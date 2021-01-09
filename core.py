#%tensorflow_version 2.x
import tensorflow as tf
keras = tf.keras
tfkl = keras.layers

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers

import gym

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

        self.zcs = tf.ones((N_SAMPLES, self.d_zc))
        self.zus = tf.zeros((N_SAMPLES, self.d_zu))

    def bottom_up(self): pass
    def top_down(self): pass
    def train(self): pass

# member functions should all be called in their declared order
class InformationNode(Node):
    """

    `f_abs: list<Tensor> [(N_samples, self.d_zc+self.d_zu),
                          (N_samples, parent1.d_zc+parent1.d_zu),
                          (N_samples, parent2.d_zc+parent2.d_zu),
                          (N_samples, parent3.d_zc+parent3.d_zu),
                          ...] -> (B: N_samples E: self.d_zc)-Dist`

    `f_pred: (N_samples, self.d_zc+self.d_zu)-Tensor -> (B: N_samples E: self.d_zc)-Dist`

    `f_act: (N_samples, self.d_zc)-Tensor -> list<Dist> [(N_samples, parent1.d_zc),
                                                         (N_samples, parent2.d_zc),
                                                         (N_samples, parent3.d_zc),
                                                         ...]`

    `f_trans: Tensor -> Dist`


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

    def __init__(self,
                 f_abs,
                 f_act,
                 f_pred,
                 d_zc=8,
                 hparams=dict()):

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
        self.child_targets = list() # N-list< [(N_samples)-Tensor, (N_samples,self.d_zc)-Dist] >

    def set_parents(self, parents):
        self.parents = parents # list<Node>

    def set_neighbors(self, neighbors):
        self.neighbors = [
            (neighbor, make_f_trans(d_zcin=neighbor.zcs.shape,
                                    d_zcout=self.zcs.shape[-1]))
            for neighbor in neighbors]

    # def set_children(self, children):
    #    self.children = {child:None for child in children}

    def bottom_up(self):
        self.xs_uncomb = [[node.zcs, node.zus] for node in [self] + self.parents]
        # (N_parents)-list< [(N_samples, node.d_zc)-Tensor, (N_samples, node.d_zu)-Tensor] >
        self.xcs = self.xs_uncomb[:,0]
        # (N_parents)-list< (N_samples, node.d_zc)-Tensor >
        self.xus = self.xs_uncomb[:,1]
        # (N_parents)-list< (N_samples, node.d_zu)-Tensor >
        self.xs = [ tf.concat([zc, zu], axis=-1) for zc, zu in self.xs_uncomb]
        # (N_parents)-list< (N_samples, node.d_zc+node.d_zu)-Tensor >
        self.record['xs'] = self.xs
        w_zabs = sum([node.w_zs for node in [self] + self.parents]) # (N_samples,)-Tensor
        # since () can be broadcast onto (N_samples,), it is okay
        # if raw sensors provide a 0-dimensional observation.
        Zcs = self.f_abs(self.xs) # (B: N_samples, E: self.d_zc)-Dist
        self.Zc = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=w_zabs),
            components_distribution=tf.unstack(Zcs, axis=0)) # (B: E: self.d_zc,)-Dist

        self.zcs = self.Zc.sample(N_SAMPLES) # (N_samples, self.d_zc)-Tensor
        if 'predictive_coding' in self.hparams and self.hparams['predictive_coding'] == True:
            self.zcs -= self.zcpreds
        self.w_zs = tf.reduce_mean(w_zabs) \
                    - self.Zc.log_prob(self.zcs) \
                    - self.Zc.entropy() # (N_samples,)-Tensor
        self.zus = tf.expand_dims(self.w_zs, axis=-1) # (N_samples, 1)-Tensor

        Zcpreds = self.f_pred(tf.concat([self.zcs, self.zus], axis=-1))
        # (B: N_samples, E: d_zc)-Dist
        Zcpred = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=self.w_zs),
            components_distribution=tf.unstack(Zcpreds, axis=0)
        ) # (B: E: self.d_zc)-Dist
        self.zcpreds = Zcpred.sample(N_SAMPLES) # (N_samples, self.d_zc)-Tensor
        self.w_zcpreds = tf.reduce_mean(self.w_zs) \
                         - Zcpred.log_prob(self.zcpreds) \
                         - Zcpred.entropy() # (N_samples,)-Tensor

    def top_down(self):
        """
        should be called after `bottom_up`
        """
        self.record['zcpred_neighbor'] = [ neighbor.zcpreds
                                           for neighbor, f_trans
                                           in self.neighbors.items()]
        Zwtargets_comb = [[neighbor.w_zcpreds, f_trans(neighbor.zcpreds)]
                          for neighbor, f_trans in self.neighbors.items()] + \
                         self.child_targets + [self.w_zcpreds, self.zcpreds]
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

        epochs = 3
        for epoch in range(epochs):
            with tf.GradientTape() as tape:

                prev_record = None
                for record in self.buffer:
                    record = { k: tf.stop_gradient(v)
                               for k, v in record.items() }

                    prev_record = record
                    if prev_record is None:
                        continue

                    ## TODO just stack timesteps into the same batch axis so the
                    #   resulting tensor is (Nsteps x Nsamples, ...) for batch processing
                    #   these tensors should be shifted by one time step so that it is
                    #   safe to treat them such a way.

                    # train f_abs, f_pred for predictability:
                    # min KL [ f_pred(f_abs(xs_prev)) || self.Z ]
                    Z_past_pred = self.f_pred(self.f_abs(prev_record['xs']))
                    Zc = self.f_abs(record['xs'])
                    L_pred = Z_past_pred.kl_divergence(Zc)

                    # train f_trans by association:
                    # min sum [ KL [ f_trans(neighbor.zcs) || self.zcs ] for neighbor in neighbors ]
                    Z_neighbors_trans = [f_trans.foreward()]

                    # train f_act: fit to psuedoinverse from f_abs
                    # ztarget_prev_samples ~ Ztarget_prev
                    # min f_act(ztarget_prev_samples).log_prob(xcs)
                    # TODO

            # TODO get gradients and minimize losses

        print('node training complete')
