#%tensorflow_version 2.x
import tensorflow as tf
keras = tf.keras
tfkl = keras.layers

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers

import gym
import time

N_SAMPLES = 3

def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return lambda t: tfd.Independent(tfd.Normal(
        loc=tf.zeros(n, dtype=dtype), scale=1),
        reinterpreted_batch_ndims=1)

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return keras.Sequential([
        tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
        tfpl.IndependentNormal(n) ])

def make_f_pred(d_zc, d_zu):

    f_pred = keras.Sequential([
        tfkl.Input(shape=(d_zc+d_zu,))
        tfpl.DenseVariational(
            units=2*d_zc,
            make_posterior_fn=posterior,
            make_prior_fn=prior,
            kl_weight=1/N,
            activation='relu'),
        tfpl.DenseVariational(
            units=2*d_zc,
            make_posterior_fn=posterior,
            make_prior_fn=prior,
            kl_weight=1/N,
            activation='relu'),
        tfpl.IndependentNormal(event_shape=(d_zc,))
    ])

    f_pred.compile(loss=lambda y_true, y_pred: -y_pred.log_prob(y_true))

    return f_pred

def make_f_trans(self, d_zcin, d_zcout):
    pass # TODO
    # input tensor
    # output distribution


class Node:

    def __init__(self, d_zc, d_zu):
        self.d_zc = d_zc
        self.d_zu = d_zu

        self.zcs = tf.ones((self.d_zc,))
        self.zus = tf.zeros((self.d_zu,))

        self.child_targets = list() # N-list< [(N_samples)-Tensor, (N_samples,self.d_zc)-Dist] >

    def bottom_up(self):
        """sensory nodes should override this function to update `self.zcs`"""
        pass
    def top_down(self):
        """actuator nodes should override this function to attempt to perform `self.child_targets`"""
        pass
    def train(self): pass


class InformationNode(Node):
    """
    member functions should all be called in their declared order

    all functions are `Tensor` -> `Dist`
    `f_abs: [(N_samples, self.d_zc),
             (N_samples, parent1.d_zc),
             (N_samples, parent2.d_zc),
             (N_samples, parent3.d_zc),
             ...,
             (N_samples, self.d_zu),
             (N_samples, parent1.d_zu),
             (N_samples, parent2.d_zu),
             (N_samples, parent3.d_zu),
             ...]
              -> (B: N_samples E: self.d_zc)`

    `f_pred: [(N_samples, self.d_zc), (self.d_zu)] -> (B: N_samples E: self.d_zc)`

    `f_act: [(N_samples, self.d_zc),  # zc
             (N_samples, self.d_zu),  # zu
             (N_samples, self.d_zc)]  # ztarget
            -> [(N_samples, parent1.d_zc),
                (N_samples, parent2.d_zc),
                (N_samples, parent3.d_zc),
                ...]`

    `f_trans: [(N_samples, neighbor.d_zc), (neighbor.d_zu)] -> (B: N_samples E: self.d_zc)`


    naming convention:
    `CAPITAL`/`lowercase`: distribution/plain-old tensor
    `-`/`-s`: single/batch

    `x`/`z`: observation (from parents)/latent (autonomous)
    `-c`/`-u`: controllable/uncontrollable

    type specification:
    `line of code returning object # SHAPE-TYPENAME<GENERIC PARAMS>`
    SHAPE and GENERIC PARAMS are optional. eg:
    `self.nodes = [node1, node2, node3] # 3-list<InformationNode>`
        means `self.nodes` is a list of 3 `InformationNode` objects
    I also use literal `[]`'s to refer to specifically enumerated element types. eg:
    `self.neighbors = list() # (N_neighbors)-list<[node,
        ((N_samples, neighbor.dzc,)-Tensor) -> (B: N_samples, E: self.dzc,)-Dist] >`
        means `self.neighbors` is an (N_neighbors x 2) list with `Node`'s in the
        first column and probabilistic functions in the second.

    I also employ `inline_var: type` specifications
    """

    NAME_COUNTER = 0

    def __init__(self,
                 f_abs,
                 f_act,
                 f_pred,
                 d_zc=8,
                 hparams=dict(),
                 name=None):

        super(InformationNode, self).__init__(d_zc=d_zc, d_zu=1)

        self.f_abs = f_abs
        self.f_act = f_act
        self.f_pred = f_pred

        # self.f_abs.compile(loss=lambda y_true, y_pred: y_pred.log_prob(y_true))
        # self.f_act.compile(loss=lambda y_true, y_pred: y_pred.log_prob(y_true))
        # self.f_pred.compile(loss=lambda y_true, y_pred: y_pred.log_prob(y_true))

        self.hparams = hparams
        self.record = dict() # `record` is progressively defined during bottom_up and top_down
        self.buffer = list() # list<tuple<?-Tensor>>

        self.parents = list() # list<Node>
        self.neighbors = dict() # dict< Node, callable>
        # callable is the controllable latent translator:
        # ((N_samples, neighbor.d_zc)-Tensor)->(N_samples, self.d_zc)-Tensor

        if name is None:
            name = f'InformationNode{InformationNode.NAME_COUNTER}'
            InformationNode.NAME_COUNTER += 1
        self.name = name

    def set_parents(self, parents):
        self.parents = parents # list<Node>

    def set_neighbors(self, neighbors):
        self.neighbors = [
            (neighbor, make_f_trans(d_zcin=neighbor.zcs.shape,
                                    d_zcout=self.zcs.shape[-1]))
            for neighbor in neighbors]

    # def set_children(self, children):
    #    self.children = {child:None for child in children}

    def build(self):
        self.w_abspred = self.f_abs.trainable_weights + self.f_pred.trainable_weights
        self.w_trans_neighbors = list()
        for _, f_trans in self.neighbors.items():
            self.w_trans_neighbors.extend(f_trans.trainable_weights)
        self.w_act = self.f_act.trainable_weights
        self.optimizer = keras.optimizers.SGD(5e-3)

    def bottom_up(self):
        self.xs_uncomb = [[node.zcs, node.zus] for node in [self] + self.parents]
        # (N_parents)-list< [(N_samples, node.d_zc)-Tensor, (N_samples, node.d_zu)-Tensor] >
        self.xcs = self.xs_uncomb[:,0]
        self.record['xcs'] = tf.stop_gradient(self.xcs)
        # (N_parents)-list< (N_samples, node.d_zc)-Tensor >
        self.xus = self.xs_uncomb[:,1]
        self.record['xus'] = tf.stop_gradient(self.xus)
        # (N_parents)-list< (N_samples, node.d_zu)-Tensor >

        #### self.xs = [ tf.concat([zc, zu], axis=-1) for zc, zu in self.xs_uncomb]
        #### # (N_parents)-list< (N_samples, node.d_zc+node.d_zu)-Tensor >

        w_zabs = sum([node.w_zs for node in [self] + self.parents]) # (N_samples,)-Tensor
        # since () can be broadcast onto (N_samples,), it is okay
        # if raw sensors provide a 0-dimensional observation.
        Zcs = self.f_abs(self.xcs + self.xus) # (B: N_samples, E: self.d_zc)-Dist
        self.Zc = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=w_zabs),
            components_distribution=tf.unstack(Zcs, axis=0)) # (B: E: self.d_zc,)-Dist

        self.zcs = self.Zc.sample(N_SAMPLES) # (N_samples, self.d_zc)-Tensor
        if 'predictive_coding' in self.hparams and self.hparams['predictive_coding'] == True:
            self.zcs -= self.zcpreds
            self.record['zcs_pred_prev'] = tf.stop_gradient(self.zcpreds)
        self.w_zs = tf.reduce_mean(w_zabs) \
                    - self.Zc.log_prob(self.zcs) \
                    - self.Zc.entropy() # (N_samples,)-Tensor
        self.zus = tf.expand_dims(self.w_zs, axis=-1) # (N_samples, 1)-Tensor
        self.record['zus'] = tf.stop_gradient(self.zus)

        Zcpreds = self.f_pred([self.zcs, self.zus])
        # (B: N_samples, E: d_zc)-Dist
        Zcpred = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=self.w_zs),
            components_distribution=tf.unstack(Zcpreds, axis=0)
        ) # (B: E: self.d_zc)-Dist
        zcpred_entropy = Zcpred.entropy() # ()-Tensor
        self.zcpreds = Zcpred.sample(N_SAMPLES) # (N_samples, self.d_zc)-Tensor
        self.zupreds = tf.expand_dims(zcpred_entropy, axis=-1) # (N_samples, 1)-Tensor
        self.w_zcpreds = tf.reduce_mean(self.w_zs) \
                         - Zcpred.log_prob(self.zcpreds) \
                         - zcpred_entropy # (N_samples,)-Tensor

    def top_down(self):
        """
        should be called after `bottom_up`
        """
        self.record['zcus_neighbors'] = [ tf.stop_gradient([neighbor.zcs, neighbor.zus])
                                           for neighbor, f_trans
                                           in self.neighbors.items()]
        Zwtargets_comb = [ [neighbor.w_zcpreds,
                            f_trans([neighbor.zcpreds, neighbor.zupreds])
                           ] for neighbor, f_trans in self.neighbors.items()] \
                         + self.child_targets \
                         + [self.w_zcpreds, self.zcpreds]
        # (N_targets)-list< (node.N_samples,)-Tensor, (B: node.N_samples, E: self.d_zc)-Dist >
        w_Ztargets = tf.stack(Zwtargets_comb[:, 0], axis=0)
        # (N_targets x node.N_samples,)-tensor
        Ztargets = tf.stack(Zwtargets_comb[:, 1], axis=0)
        # (B: N_targets x node.N_samples, E:self.d_zc)-Dist
        self.Ztarget = tfd.Mixture(
            cat=tfd.Categorical(logits=w_Ztargets),
            components=tf.unstack(Ztargets, axis=0)
        )
        # (B: E: self.d_zc,)-dist
        ztargets = self.Ztarget.sample(N_SAMPLES)
        self.record['ztargets'] = tf.stop_gradient(ztargets)
        # (N_samples, self.d_zc)-Tensor
        Xparentstargets = self.f_act([self.zcs, self.zus, ztargets])
        # (N_parents)-list < (B: N_samples, E: parent.d_zc)-Dist >
        w_Xtargets = tf.reduce_mean(self.w_zs) - self.Ztarget.log_prob(ztargets)
        # (N_samples,)-Tensor
        for parent, Xparenttargets in zip(self.parents, Xparentstargets):
            parent.child_targets.append([w_Xtargets - Xparenttargets.entropy(), Xparenttargets])

    def record_step(self):
        self.buffer.append(self.record)
        self.record = dict()

    def train(self):
        """train on most recently observed data.
        should be called after `top_down`
        """

        # NOTE since these are RV's, multiple epochs can
        # glean considerable improvement with little data
        epochs = 5
        for epoch in range(epochs):

            print(f'Beginning epoch:{epoch} at {time}')

            L_abspred = 0.
            L_trans_neighbors = 0.
            L_act = 0.

            with tf.GradientTape() as tape:

                prev_record = None
                for record in self.buffer:
                    prev_record = record
                    if prev_record is None:
                        continue

                    # TODO just stack timesteps into the same batch axis so the
                    #   resulting tensor is (Nsteps x Nsamples, ...) for batch processing
                    #   These tensors are already shifted by one time step so it is safe
                    #   to treat them such a way. If I do stack timesteps, I should only
                    #   stack the batches in subgroups so the 'Predictive' part of `Node`
                    #   still has the opportunity to observe its own predictions in training
                    #   eg: reorganize into batches from the following timesteps: [1,11,21,31],
                    #   [2,12,22,32],...[9,19,29,39].

                    # TODO: the variable nomenclature has since changed and I should update
                    #   the comments in psuedocode-math to reflect this. I also need to add shape
                    #   and type specifications after each statement

                    # train f_abs, f_pred for predictability:
                    # min KL [ f_pred(f_abs(xs_prev)) || self.Z ]
                    Zcs_prev = self.f_abs(prev_record['xcs'] + prev_record['xus'], training=True)
                    if 'predictive_coding' in self.hparams and self.hparams['predictive_coding'] == True:
                        Zcs_prev = tfb.Affine(shifts=-prev_record['zcs_pred_prev']).forward(Zcs_prev)
                        # NOTE: since prev_record['xcs'] came in sample-wise divisions,
                        # it makes sense to continue using those divisions for the
                        # predictive coding subtraction. I will have to write a
                        # custom Bijector to use rectified subtraction
                    zus_prev = tf.expand_dims(Zcs_prev.entropy(), axis=-1)
                    Zc_past_preds = self.f_pred([Zcs_prev, zus_prev])
                    Zcs = self.f_abs(record['xcs'] + record['xus'], training=True)
                    if 'predictive_coding' in self.hparams and self.hparams['predictive_coding'] == True:
                        Zcs = tfb.Affine(shifts=-record['zcs_pred_prev']).forward(Zcs)
                        # NOTE: since record['xcs'] came in sample-wise divisions,
                        # it makes sense to continue using those divisions for the
                        # predictive coding subtraction. I will have to write a
                        # custom Bijector to use rectified subtraction
                    L_abspred = L_abspred + Zc_past_preds.kl_divergence(Zcs)

                    # train f_trans by association:
                    # min sum [ KL [ f_trans(neighbor.zcs) || self.zcs ] for neighbor in neighbors ]
                    Zcs_neighbors = tf.stack([f_trans(zcus_neighbor)
                                              for f_trans, zcus_neighbor
                                              in zip(list(self.neighbors.values()),
                                                     record['zcus_neighbors'])
                                             ], axis=0)
                    L_trans_neighbors = L_trans_neighbors + Zcs_neighbors.kl_divergence(Zcs)

                    # train f_act: fit to psuedoinverse from f_abs
                    # ztarget_prev_samples ~ Ztarget_prev
                    # min f_act(ztarget_prev_samples).log_prob(xcs)
                    Xparentstargets_prev = self.f_act([Zcs_prev, zus_prev, prev_record['ztargets']])
                    L_act = L_act + sum([Xparenttargets_prev.log_prob(parent_xcs)
                                         for Xparenttargets_prev, parent_xcs
                                         in zip(Xparentstargets_prev, record['xcs'])])
                    # also treat f_act as an inverse learner (assuming Zcs_next ~= Ztarget)
                    # min Err( xcs , f_act(Zcs_prev, zus_prev, Zcs) )
                    # I actually use log_prob rather than MSE or algebraic error metrics
                    Xparentstargets_prev = self.f_act([Zcs_prev, zus_prev, Zcs]) # these are `ConvertToTensor`'s
                    L_act = L_act + sum([Xparenttargets_prev.log_prob(parent_xcs)
                                         for Xparenttargets_prev, parent_xcs
                                         in zip(Xparentstargets_prev, record['xcs'])])

            # get gradients and minimize losses
            grad_abspred = tape.gradient(L_abspred, self.w_abspred)
            grad_trans_neighbors = tape.gradient(L_trans_neighbors, self.w_trans_neighbors)
            grad_act = tape.gradient(L_act, self.w_act)

            self.optimizer.apply_gradients(zip(grad_abspred, self.w_abspred))
            self.optimizer.apply_gradients(zip(grad_trans_neighbors, self.w_trans_neighbors))
            self.optimizer.apply_gradients(zip(grad_act, self.w_act))

        print(f'{self.name} training complete')
