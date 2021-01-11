#%tensorflow_version 2.x
import tensorflow as tf
keras = tf.keras
tfkl = keras.layers
K = keras.backend

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers

import gym
import time
import math
import logging

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

    NAME_COUNTER = 1

    def __init__(self, name=None):
        if name is None:
            name = f'node_{Node.NAME_COUNTER}'
            Node.NAME_COUNTER += 1
        self.name = name


class UncontrollableNode(Node):

    def __init__(self, d_zu, **kwargs):

        super(UncontrollableNode, self).__init__(**kwargs)

        self.d_zu = d_zu

        self.zus = tf.zeros((N_SAMPLES, self.d_zu,))
        self.w_zs = tf.zeros((N_SAMPLES,))

    def reset_states(self):
        """called before beginning a new episode or training on collected data"""
        self.zus = tf.zeros_like(self.zus)
        self.w_zs = tf.zeros(self.w_zs)


class ControllableNode(Node):

    def __init__(self, d_zc, **kwargs):

        super(ControllableNode, self).__init__(**kwargs)
        self.d_zc = d_zc

        self.zcs = tf.zeros((N_SAMPLES, self.d_zc,))
        self.w_zs = tf.zeros((N_SAMPLES,))
        self.child_targets = list() # N-list< [(N_samples)-Tensor, (N_samples,self.d_zc)-Dist] >

    def reset_states(self):
        """called before beginning a new episode or training on collected data"""
        self.zcs = tf.zeros_like(self.zcs)
        self.w_zs = tf.zeros(self.w_zs)
        self.child_targets.clear()


class SensoryNode(UncontrollableNode):

    def update_state(self, observation):
        """sensory nodes should override this function to update `self.zcs`"""
        raise NotImplementedError()


class ActuatorNode(ControllableNode):

    def get_action(self):
        """actuator nodes should override this function to attempt to perform `self.child_targets`"""
        raise NotImplementedError()


class LCA(tfkl.Layer):

    def __init__(self,
                 N_w,
                 lr_w=1e-3,
                 winner_take_all=True,
                 pool_act_axes=[],
                 backpropagatable=False):
        super(LCA, self).__init__()
        self.N_w = N_w
        self.lr_w = lr_w
        self.winner_take_all = winner_take_all
        self.pool_act_axes = pool_act_axes
        self.backpropagatable = backpropagatable

    def build(self, input_shape):
        d_x = input_shape[-1] # (1,)-Tensor

        self.ws = tf.random.uniform(shape=(d_x, self.N_w),
                                    minval=self.minval,
                                    maxval=self.maxval) # ws: (d_x, N_w)
        if self.backpropagatable:
            self.ws = tf.Variable(self.ws, trainable=True)

        self.ws = tf.Variable()

        if not self.winner_take_all:
            self.batch_norm_layer = tfkl.BatchNormalization(axis=-2)

    def call(self, inputs, **kwargs):

        # inputs: (..., d_x)
        X = tf.expand_dims(inputs, axis=-1) # (..., d_x, 1)
        dist = X - self.ws # (..., d_x, N_w)
        act = tf.exp(-dist**2) # (..., d_x, N_w)

        if self.winner_take_all:
            ind_ws = K.argmax(act, axis=-1) # (..., 1)
            act = tf.one_hot(ind_ws, depth=self.N_w, axis=-1) # (..., d_x, N_w)
        else:
            act = self.batch_norm_layer(act) # (..., d_x, N_w)

        if 'training' in kwargs and kwargs['training']:
            # move weights closer to inputs
            dws = tf.sign(dist) * act # (..., d_x, N_w)
            sum_axes = tf.range(tf.rank(dws)-3) # (?,)
            # eg: if `dws.shape=(B, N_x, d_x, N_w)`, then `rank(dws)=4`,
            # `sum_axes=tf.range(2)=[0,1]`.
            dws = tf.reduce_sum(dws, axis=sum_axes) # (d_x, N_w)
            self.ws = self.ws + self.lr_w * dws # (d_x, N_w)

        if len(self.pool_act_axes) > 0:
            act = K.sum(act, axis=self.pool_act_axes)
            # eg: for an image, `X.shape=(B, N_y, N_x, N_c, N_w)`
            # - if `self.pool_act_axes=[-2]`, this performs 1D convolution
            # - if `self.pool_act_axes=[-4,-3,-2]`, this performs image feature pooling
            # NOTE: `self.pool_act_axes` should not include the batch axis

        return act # (..., N_w) or some dimensional reduction therefrom


class VisionNode(SensoryNode):

    def __init__(self, d_zu, enc=None):
        super(VisionNode, self).__init__(d_zc=0, d_zu=d_zu)

        if enc is None:
            enc = LCA(N_w=64, pool_act_axes=[-4,-3,-2])
        self.enc = enc

    def update_state(self, observation):
        self.zcs = self.enc(observation, trainable=True)


class PredictionNode(UncontrollableNode, ControllableNode):
    """
    Similar to a node in a Bayesian network, an `PredictionNode` has parents forming
    its receptive field, children that it directly influences, and neighbors that may
    or may not share the same parents.

    Internally, `PredictionNode` builds a recurrent latent representation of its receptive
    field for next-frame latent representation prediction accuracy. It then acts to fulfill
    the predictions it makes by sending expected observation targets to Nodes in its receptive
    field. It also attempts to fulfill the targets from `Node`'s that it forms the receptive
    field for. It also takes into consideration the predictions of its neighbors when forming
    its own predictions and gives more attention to minimum entropy target states.

    To prevent zero-representation collapse, `f_abs` should also include some
    unsupervised bottom-up representation mechanisms which the latent state is only
    able to modify but not completely ignore. Maybe `f_abs` is a latent to latent bijector
    parametrized by the unsupervised features. Optionally, the anticipated representation
    is subtracted from what is actually formed if `predictive_coding` is enabled.


    Functional Structure:
    All functions are `Tensor` -> `Dist`
    These functions' `Dist` outputs are fully recognized as such and not
    treated as `Tensor`'s until calling `.sample`. This means they can have
    `tfpl.Layer`'s output `Dist`'s propagated through `Bijector`'s in their
    final layers without issues.

    `f_abs: [self.zcs:    (N_samples, self.d_zc),
             parent1.zcs: (N_samples, parent1.d_zc),
             parent2.zcs: (N_samples, parent2.d_zc),
             parent3.zcs: (N_samples, parent3.d_zc),
             ...,
             self.zus:    (N_samples, self.d_zu),
             parent1.zus: (N_samples, parent1.d_zu),
             parent2.zus: (N_samples, parent2.d_zu),
             parent3.zus: (N_samples, parent3.d_zu),
             ...]
            -> self.Zcs: (B: N_samples E: self.d_zc)`

    `f_pred: [self.zcs: (N_samples, self.d_zc),
              self.zus: (self.d_zu)]
             -> self.Zcpreds (B: N_samples E: self.d_zc)`

    `f_act: [self.zcs: (N_samples, self.d_zc),
             self.zus: (N_samples, self.d_zu),
             ztargets: (N_samples, self.d_zc)]
            -> [ztargets for parent1: (N_samples, parent1.d_zc),
                ztargets for parent2: (N_samples, parent2.d_zc),
                ztargets for parent3: (N_samples, parent3.d_zc),
                ...]`

    `f_trans: [neighbor.zcs: (N_samples, neighbor.d_zc),
               neighbor.zus: (neighbor.d_zu)]
              -> zcs (translated): (B: N_samples E: self.d_zc)`

    Although `f_act` outputs a list of independent `Distributions`, they are really
    dependant. This should be modeled by using a/some `ConvertToTensor` `tfpl.Layer` in
    the bottleneck layers of `f_act`

    Naming Convention:
    `CAPITAL`/`lowercase`: distribution / real-valued number
    `x`/`z`: observation (from parents) / latent (internal state)
    `-c`/`-u`: controllable / uncontrollable
    `-`/`-s`: single / sample batch

    Type Specification Syntax:
    `Line of code returning object # SHAPE-TYPENAME<GENERIC PARAMS>`
    and in pseudocode `obj: SHAPE-TYPENAME<GENERIC PARAMS>`.
    SHAPE and GENERIC PARAMS are optional. eg:
    `self.nodes = [node1, node2, node3] # 3-list<PredictionNode>`
        means `self.nodes` is a list of 3 `PredictionNode` objects
    I also use literal `[]`'s to refer to specifically enumerated element types. eg:
    `self.neighbors = list() # (N_neighbors)-list<[node,
        ((N_samples, neighbor.dzc,)-Tensor) -> (B: N_samples, E: self.dzc,)-Dist]>`
        means `self.neighbors` is an (N_neighbors x 2) list with `Node`'s in the
        first column and probabilistic functions in the second.
    I also employ `inline_var: type` specifications
    For conciseness, I abbreviate `tfp.distributions.Distribution` as `Dist`

    `PredictionNode` member functions should all be called in their declared order
    """

    def __init__(self,
                 f_abs,
                 f_act,
                 f_pred,
                 d_zc=8,
                 hparams=dict(),
                 name=None):

        super(PredictionNode, self).__init__(d_zc=d_zc, d_zu=1, name=name)

        self.f_abs = f_abs
        self.f_act = f_act
        self.f_pred = f_pred

        self.parents = list() # list<Node>
        self.neighbors = dict() # dict< Node, callable>
        # callable is the controllable latent translator:
        # ((N_samples, neighbor.d_zc)-Tensor)->(N_samples, self.d_zc)-Tensor

        self.hparams = hparams
        self.record = dict() # `record` is progressively defined during bottom_up and top_down
        self.buffer = list() # list<dict<str, Tensor>>

        self.predictive_coding = 'predictive_coding' in self.hparams \
                                 and self.hparams['predictive_coding']

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

    def reset_states(self):
        super(PredictionNode, self).reset_states()
        # TODO reset other internal state variables
        #   use `tf.zeros_like` for simplicity
        self.zcpreds = tf.zeros_like(self.zcpreds)
        self.w_zcpreds = tf.zeros_like(self.w_zcpreds)
        self.record.clear()

    def bottom_up(self):
        xs_uncomb = [[node.zcs, node.zus] for node in self.parents]
        # (N_parents)-list< [(N_samples, node.d_zc)-Tensor, (N_samples, node.d_zu)-Tensor] >
        xcs = xs_uncomb[:,0]
        self.record['xcs'] = tf.stop_gradient(xcs)
        # (N_parents)-list< (N_samples, node.d_zc)-Tensor >
        xus = xs_uncomb[:,1]
        self.record['xus'] = tf.stop_gradient(xus)
        # (N_parents)-list< (N_samples, node.d_zu)-Tensor >
        w_zabs = sum([node.w_zs for node in [self] + self.parents]) # (N_samples,)-Tensor
        self.record['w_zabs'] = tf.stop_gradient(w_zabs)
        # since () can be broadcast onto (N_samples,), it is okay
        # if raw sensors provide a 0-dimensional observation.

        self._bottom_up(xcs=xcs, xus=xus, w_zabs=w_zabs)

    def _bottom_up(self, xcs, xus, w_zabs):
        Zcs = self.f_abs([self.zcs] + xcs + [self.zus] + xus)
        # (B: N_samples, E: self.d_zc)-Dist
        self.Zc = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=w_zabs),
            components_distribution=Zcs
        ) # (B: E: self.d_zc,)-Dist

        self.zcs = self.Zc.sample(N_SAMPLES) # (N_samples, self.d_zc)-Tensor
        if self.predictive_coding:
            self.zcs = keras.activations.relu(self.zcs-self.zcpreds)
        self.w_zs = tf.reduce_mean(w_zabs) \
                    - self.Zc.log_prob(self.zcs) \
                    - self.Zc.entropy() # (N_samples,)-Tensor
        self.zus = tf.expand_dims(self.w_zs, axis=-1) # (N_samples, 1)-Tensor

        Zcpreds = self.f_pred([self.zcs, self.zus]) # (B: N_samples, E: d_zc)-Dist
        self.Zcpred = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=self.w_zs),
            components_distribution=Zcpreds
        ) # (B: E: self.d_zc)-Dist
        self.zcpreds = self.Zcpred.sample(N_SAMPLES) # (N_samples, self.d_zc)-Tensor
        self.w_zcpreds = tf.reduce_mean(self.w_zs) \
                         - self.Zcpred.log_prob(self.zcpreds) \
                         - self.Zcpred.entropy() # (N_samples,)-Tensor
        self.zupreds = tf.expand_dims(self.w_zcpreds, axis=-1) # (N_samples, 1)-Tensor

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
        # (N_targets)-list<[(node.N_samples,)-Tensor, (B: node.N_samples, E: self.d_zc)-Dist]>
        w_Ztargets = tf.concat(Zwtargets_comb[:, 0], axis=0)
        # (N_targets x node.N_samples,)-Tensor
        Ztargets = tf.unstack(Zwtargets_comb[:, 1], axis=0)
        # (N_targets x node.N_samples)-list<(B: E: self.d_zc)-Dist>
        self.Ztarget = tfd.Mixture(
            cat=tfd.Categorical(logits=w_Ztargets),
            components=tf.unstack(Ztargets, axis=0)
        )
        # (B: E: self.d_zc,)-Dist
        ztargets = self.Ztarget.sample(N_SAMPLES)
        self.record['zcs'] = tf.stop_gradient(self.zcs) # NOTE: these entries are to only be used
        self.record['zus'] = tf.stop_gradient(self.zus) # NOTE: for `f_act` training (not `f_pred`)
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
        t_trainstart = time.time()
        logging.log(f'beginning {self.name} training at {round(t_trainstart, 3)}')

        # NOTE since these are RV's, multiple epochs can
        # glean considerable improvement with little data
        epochs = 5
        for epoch in range(epochs):

            t_epochstart = time.time()
            logging.log(f'Beginning epoch:{epoch} at {round(t_epochstart, 3)}')

            self.reset_states()

            L_abspred = 0.
            L_trans_neighbors = 0.
            L_act = 0.
            Zcpred_prev = None

            with tf.GradientTape() as tape:

                prev_record = None
                for record in self.buffer:

                    self._bottom_up(xcs=record['xcs'],
                                    xus=record['xus'],
                                    w_zabs=record['w_zabs'])

                    Zcpred_prev = self.Zcpred
                    if prev_record is None:
                        prev_record = record
                        continue
                    else:
                        prev_record = record

                    # train f_abs, f_pred for predictability:
                    # min KL [ f_pred(f_abs(...[prev step]...)) || self.Z ]
                    L_abspred = L_abspred + tf.reduce_sum(Zcpred_prev.kl_divergence(self.Zc))

                    # train f_trans by association:
                    # min sum [ KL [ f_trans(neighbor.zcs) || self.zcs ] for neighbor in neighbors ]
                    Zcs_neighbors = tf.stack([f_trans(zcus_neighbor)
                                              for f_trans, zcus_neighbor
                                              in zip(list(self.neighbors.values()),
                                                     record['zcus_neighbors'])
                                             ], axis=0)
                    # (B: N_neighbors, N_samples, E: self.d_zc)-Dist
                    # TODO make sure `kl_divergence` can broadcast over 2 batch axes
                    L_trans_neighbors = L_trans_neighbors + tf.reduce_sum(Zcs_neighbors.kl_divergence(self.Zcs))

                    # train f_act to fit to 'psuedoinverse' of f_abs
                    # min f_act(zcs_prev, zus_prev, ztargets_prev).log_prob(xcs)
                    Xparentstargets_prev = self.f_act([prev_record['zcs'],
                                                       prev_record['zus'],
                                                       prev_record['ztargets']])
                    L_act = L_act + tf.reduce_sum(sum([Xparenttargets_prev.log_prob(parent_xcs)
                                                       for Xparenttargets_prev, parent_xcs
                                                       in zip(Xparentstargets_prev, record['xcs'])]))

                    # also treat f_act as an inverse learner (assuming Zcs_next ≈ Ztarget)
                    # min f_act(zcs_prev, zus_prev, zcs).log_prob(xcs)
                    Xparentstargets_prev = self.f_act([prev_record['zcs'],
                                                       prev_record['zus'],
                                                       record['zcs']])
                    L_act = L_act + tf.reduce_sum(sum([Xparenttargets_prev.log_prob(parent_xcs)
                                                       for Xparenttargets_prev, parent_xcs
                                                       in zip(Xparentstargets_prev, record['xcs'])]))

            # get gradients and minimize losses
            logging.log(f'Getting gradients. Time elapsed: {round(time.time() - t_epochstart, 3)}s')
            grad_abspred = tape.gradient(L_abspred, self.w_abspred)
            grad_trans_neighbors = tape.gradient(L_trans_neighbors, self.w_trans_neighbors)
            grad_act = tape.gradient(L_act, self.w_act)

            logging.log(f'Applying gradients. Time elapsed: {round(time.time()-t_epochstart, 3)}s')
            self.optimizer.apply_gradients(zip(grad_abspred, self.w_abspred))
            self.optimizer.apply_gradients(zip(grad_trans_neighbors, self.w_trans_neighbors))
            self.optimizer.apply_gradients(zip(grad_act, self.w_act))

            logging.log(f'Epoch completed. Time elapsed: {round(time.time() - t_epochstart, 3)}s\n'+ 32*'-')

        logging.log(f'{self.name} training complete. Time elapsed: {round(t_trainstart, 3)}s\n'+ 32*'=')
        self.buffer.clear()


class Organism:

    def __init__(self, prediction_nodes, sensory_nodes, actuator_nodes):
        """predictions nodes should've been pre-structured"""
        self.prediction_nodes = prediction_nodes
        self.sensory_nodes = sensory_nodes
        self.actuator_nodes = actuator_nodes

    def act(self, observations):
        """recieves and returns dict<str,any> structured observations and actions"""
        self.write_sensors(observations)
        self.thinking_step()
        return self.read_actuators()

    def write_sensors(self, observations):
        for sensory_node in self.sensory_nodes:
            sensory_node.update_state(observations[sensory_node.name])

    def read_actuators(self):
        return { actuator_node.name: actuator_node.get_action
                 for actuator_node in self.actuator_nodes }

    def thinking_step(self):
        for prediction_node in self.prediction_nodes:
            prediction_node.bottom_up()
        for prediction_node in self.prediction_nodes:
            prediction_node.top_down()
