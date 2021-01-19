from .lca import SemisupervisedLCA

import tensorflow as tf
import tensorflow_probability as tfp
import time
import random
import logging

keras = tf.keras
tfkl = keras.layers
K = keras.backend

tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers

N_SAMPLES = 3


def make_abs_act_pair():

    def prior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        return lambda t: tfd.Independent(tfd.Normal(
            loc=tf.zeros(n, dtype=dtype), scale=1),
            reinterpreted_batch_ndims=1)

    def posterior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        return keras.Sequential([
            tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
            tfpl.IndependentNormal(n)])

    class DynamicActor(tfkl.Layer):
        def __init__(self, **kwargs):
            super(DynamicActor, self).__init__(**kwargs)

            self.parent_nets = list()

        def build(self, input_shape):
            self.d_zc = input_shape[0][-1]
            self.d_zu = input_shape[1][-1]

            self.foreward_model = keras.Sequential([
                tfkl.Dense(self.d_zc + self.d_zu, activation='relu'),
                tfpl.DenseVariational(self.d_zc,
                                      make_posterior_fn=posterior,
                                      make_prior_fn=prior,
                                      kl_weight=1 / 256,
                                      activation='relu'),
                tfkl.Dense(self.d_zc, activation='relu')
            ])

        def call(self, inputs, **kwargs):
            inpts_concat = tfkl.concatenate(inputs)
            hidden = self.foreward_model(inpts_concat)
            return [parent_net(hidden) for parent_net in self.parent_nets]

        def set_parent_shapes(self, shapes):
            for shape in shapes:
                def parent_target(x):
                    x = tfkl.Dense(shape[-1], activation='relu')(x)
                    loc = tfkl.Dense(shape[-1], activation='relu')(x)
                    scale = tfkl.Dense(shape[-1], activation='relu')(x)
                    return tfd.Normal(loc=loc, scale=scale)
                self.parent_nets.append(parent_target)

    class DynamicAbstractor(tfkl.Layer):

        def __init__(self, dynamic_actor, **kwargs):
            super(DynamicAbstractor, self).__init__(**kwargs)
            self.dynamic_actor = dynamic_actor

        def build(self, input_shape):
            N_parents = (len(input_shape) // 2) - 1
            self.d_zc = input_shape[0]
            self.d_zu = input_shape[N_parents+1]

            self.dynamic_actor.set_parent_shapes(input_shape[1:N_parents]+input_shape[N_parents+1:])

            zc_inpts = list()
            zu_inpts = list()
            parents = list()
            for i in range(1,1+N_parents):
                d_zci = input_shape[i][-1]
                d_zui = input_shape[1+N_parents+i][-1]
                zci_inpt = tfkl.Input((d_zci,))
                zui_inpt = tfkl.Input((d_zui,))
                zi_cat = tfkl.Concatenate()([zci_inpt, zui_inpt])
                hidi1 = SemisupervisedLCA(d_supervised=d_zci//2, d_unsupervised=d_zci//2, N_splits=4)(zi_cat)
                hidi2 = SemisupervisedLCA(d_supervised=self.d_zc//2, d_unsupervised=self.d_zc//2, N_splits=4)(hidi1)

                zc_inpts.append(zci_inpt)
                zu_inpts.append(zui_inpt)
                parents.append(hidi2)

            parents_combined = tfkl.Add(name=f'{self.name}_parents_combined')(parents)

            zcself_inpt = tfkl.Input((self.d_zc,))
            zuself_inpt = tfkl.Input((self.d_zu,))
            zself_cat = tfkl.Concatenate()([zcself_inpt, zuself_inpt])

            N_heads = 4
            d_key = self.d_zc // 2
            d_val = d_key * 2
            make_transformer_compatible = tfkl.Lambda(lambda x: x[...,None,:])
            qs = [tfkl.Dense(d_key, activation='relu')(make_transformer_compatible(zself_cat))
                  for _ in range(N_heads)]
            vs = [tfkl.Dense(d_val, activation='relu')(make_transformer_compatible(parents_combined))
                  for _ in range(N_heads)]
            ks = [tfkl.Dense(d_key, activation='relu')(make_transformer_compatible(parents_combined))
                  for _ in range(N_heads)]
            attended_vals = [tfkl.Attention(use_scale=True)([q,v,k])
                             for q, v, k in zip(qs, vs, ks)]
            final_cat = tfkl.Concatenate()([attended_vals, zself_cat])
            final1 = SemisupervisedLCA(d_supervised=self.d_zc//2,
                                       d_unsupervised=self.d_zc-(self.d_zc//2),
                                       N_splits=4)(final_cat)

            self.model = keras.Model(
                inputs=[zcself_inpt]+zc_inpts+[zuself_inpt]+zu_inpts,
                outputs=final1
            )

        def call(self, inputs, **kwargs):
            return self.model(inputs=inputs, **kwargs)

    dynamic_actor = DynamicActor()
    dynamic_abstractor = DynamicAbstractor(dynamic_actor=dynamic_actor)
    return dynamic_abstractor, dynamic_actor


def make_tensor_to_dist(d_zcin, d_zuin, d_zcout):

    zcs_inpt = tfkl.Input((d_zcin,))
    zus_inpt = tfkl.Input((d_zuin,))
    hid0 = tfkl.Concatenate()([zcs_inpt, zus_inpt])
    hidA1 = tfkl.Dense(d_zcin//2, activation='relu')(hid0)
    hidA2 = tfkl.Dense(tfpl.IndependentNormal.params_size(event_shape=(d_zcin//4,)), activation='relu')(hidA1)
    hidA3 = tfpl.IndependentNormal(event_shape=(d_zcin//4,))(hidA2)

    hidB1 = tfkl.Concatenate()([hid0, hidA3])
    hidB2 = tfkl.Dense(d_zcout, activation='relu')(hidB1)
    loc = tfkl.Dense(d_zcout, activation='relu')(hidB2)
    scale = tfkl.Dense(d_zcout, activation='relu')(hidB2)

    output = tfpl.DistributionLambda(
        make_distribution_fn=lambda locscale:
            tfd.Normal(loc=locscale[0], scale=locscale[1]))([loc, scale])

    return keras.Model(inputs=[zcs_inpt, zus_inpt], outputs=[output])


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
        self.w_zs = tf.ones((N_SAMPLES,))

    def reset_states(self):
        """called before beginning a new episode or training on collected data"""
        self.zus = tf.zeros((N_SAMPLES, self.d_zu,))
        self.w_zs = tf.ones((N_SAMPLES,))


class ControllableNode(Node):

    def __init__(self, d_zc, **kwargs):

        super(ControllableNode, self).__init__(**kwargs)
        self.d_zc = d_zc

        self.zcs = tf.zeros((N_SAMPLES, self.d_zc,))
        self.w_zs = tf.ones((N_SAMPLES,))
        self.child_targets = [[
            self.w_zs, tfd.Normal(loc=self.zcs, scale=1e-1)
        ]] # N-list< [(N_samples)-Tensor, (N_samples,self.d_zc)-Dist] >

    def reset_states(self):
        """called before beginning a new episode or training on collected data"""
        self.zcs = tf.zeros((N_SAMPLES, self.d_zc,))
        self.w_zs = tf.ones((N_SAMPLES,))
        self.child_targets = [[
            self.w_zs, tfd.Normal(loc=self.zcs, scale=1e-1)
        ]] # N-list< [(N_samples)-Tensor, (N_samples,self.d_zc)-Dist] >


class SensoryNode(ControllableNode, UncontrollableNode):
    # TODO make SensoryNode `have` nodes but not be a node

    def __init__(self, **kwargs):
        super(SensoryNode, self).__init__(**kwargs)
        self.reset_states()

    def update_state(self, observation):
        """sensory nodes should override this function to update `self.zcs`"""
        raise NotImplementedError()


class ActuatorNode(ControllableNode, UncontrollableNode):
    # TODO make ActuatorNode `have` nodes but not be a node

    def __init__(self, **kwargs):
        super(ActuatorNode, self).__init__(**kwargs)
        self.reset_states()

    def get_action(self):
        """actuator nodes should override this function to attempt to perform `self.child_targets`"""
        raise NotImplementedError()


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
              self.zus: (N_samples, self.d_zu)]
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
                 f_abs=None,
                 f_act=None,
                 f_pred=None,
                 d_zc=16,
                 predictive_coding=True,
                 name=None):

        super(PredictionNode, self).__init__(d_zc=d_zc, d_zu=1, name=name)
        #self.reset_states()

        if f_abs is None and f_act is None:
            f_abs, f_act = make_abs_act_pair()

        if f_pred is None:
            f_pred = make_tensor_to_dist(d_zcin=self.d_zc, d_zuin=self.d_zu, d_zcout=self.d_zc)

        self.f_abs = f_abs
        self.f_act = f_act
        self.f_pred = f_pred

        self.parents = list() # list<Node>
        self.neighbors = list() # list<[Node, callable]>
        # callable is the controllable latent translator:
        # ((N_samples, neighbor.d_zc)-Tensor)->(N_samples, self.d_zc)-Tensor

        self.record = dict() # `record` is progressively defined during bottom_up and top_down
        self.buffer = list() # list<dict<str, Tensor>>

        self.predictive_coding = predictive_coding

        self._is_built = False

    def add_parents(self, parents):
        self.parents.extend(parents)

    def add_neighbors(self, neighbors):
        self.neighbors.extend([
            (neighbor, make_tensor_to_dist(
                d_zcin=neighbor.d_zc, d_zuin=neighbor.d_zu, d_zcout=self.d_zc))
            for neighbor in neighbors])

    # def set_children(self, children):
    #    self.children = {child:None for child in children}

    def build(self):
        if self._is_built:
            return
        self.parents = list(set(self.parents))
        self.neighbors = list(set(self.neighbors))
        self.w_abspred = self.f_abs.trainable_weights + self.f_pred.trainable_weights
        self.w_trans_neighbors = list()
        for elem in self.neighbors:
            _, f_trans = elem
            self.w_trans_neighbors.extend(f_trans.trainable_weights)
        self.w_act = self.f_act.trainable_weights
        self.optimizer = keras.optimizers.SGD(5e-3)
        self._is_built = True

    def reset_states(self):
        super(PredictionNode, self).reset_states()
        self.zcpreds = tf.zeros_like(self.zcs)
        self.w_zcpreds = tf.zeros_like(self.w_zs)
        self.record.clear()

    def bottom_up(self):
        xcs = [node.zcs for node in self.parents]
        self.record['xcs'] = tf.stop_gradient(xcs)
        # (N_parents)-list< (N_samples, node.d_zc)-Tensor >
        xus = [node.zcs for node in self.parents]
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
            with tf.GradientTape() as tape:

                L_abspred = 0.
                L_trans_neighbors = 0.
                L_act = 0.
                Zcpred_prev = None
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

