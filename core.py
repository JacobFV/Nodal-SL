#%tensorflow_version 2.x
import tensorflow as tf
keras = tf.keras
tfkl = keras.layers

import tensorflow_probability as tfp
tfd = tfp.distributions
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
    # make bijector


class Node:

    def __init__(self, d_zc, d_zu):
        self.d_zc = d_zc
        self.d_zu = d_zu

        self.zc = tf.ones((N_SAMPLES, self.d_zc))
        self.zu = tf.zeros((N_SAMPLES, self.d_zu))

    def bottom_up(self): pass
    def top_down(self): pass
    def train(self): pass

# member functions should all be called in their declared order
class InformationNode(Node):
    """



    naming convention:
    CAPITAL/lowercase: distribution/plain-old tensor
    -/-s: single/batch

    x/z: observation (from parents)/latent (autonomous)
    -c/-u: controllable/uncontrollable

    type specification:
    line of code returning object # SHAPE-TYPENAME<GENERIC PARAMS>[CONTENTS]
    SHAPE and GENERIC PARAMS are optional
    eg:
    self.nodes = [node1, node2, node3] # 3-list<InformationNode>[A_Node, B_Node, A_Node]
        means `self.nodes` is a list of 3 `InformationNode` objects where the 0th and
        2nd items are `A_Nodes` and the 1st item is a `B_Node`.
    I also employ `inline_var: type` specifications
    """

    def __init__(self,
                 f_abs,
                 f_act,
                 f_pred,
                 d_zc=8):

        super(InformationNode, self).__init__(d_zc=d_zc, d_zu=1)

        self.f_abs = f_abs
        self.f_act = f_act
        self.f_pred = f_pred

        self.parents = list() # list<Node>
        self.neighbors = dict() # dict< Node, callable>
        # callable is the controllable latent translator:
        # ((N_samples, neighbor.d_zc)-Tensor)->(N_samples, self.d_zc)-Tensor
        self.child_targets = list() # N-list< 2-list<Tensor>[(N_samples)-Tensor, (N_samples,self.d_zc)-Dist] >

    def set_parents(self, parents):
        self.parents = parents # list<Node>

    def set_neighbors(self, neighbors):
        self.neighbors = [
            (neighbor, make_f_trans(d_zcin=neighbor.zc.shape,
                                    d_zcout=self.zc.shape[-1]) )
            for neighbor in neighbors]

    # def set_children(self, children):
    #    self.children = {child:None for child in children}

    def bottom_up(self):
        parent_inputs = [tf.concat([parent.zc, parent.zu], axis=-1)
                        for parent in self.parents]
        # (N_parents+1)-List< (N_samples, node.d_zc+node.d_zu)-Tensor >
        w_zabs = sum([node.w_z for node in [self] + self.parents]) # (N_samples,)-Tensor
        Zcs = self.f_abs([self.zc, self.zu] + parent_inputs) # (B: N_samples, E: d_zc)-Dist
        self.Zc = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=w_zabs),
            components_distribution=tf.unstack(Zcs, axis=0)) # (B: E: d_zc,)-Dist

        self.zc = self.Zc.sample(N_SAMPLES) # (B: N_samples, E: d_zc)-Dist
        self.w_z = -(self.Zc.log_prob(self.zc)+self.Zc.entropy()) # (N_samples,)-Tensor
        tf.assert_rank(self.w_z, 1, '`self.w_z` should be (N_SAMPLES)')
        self.zu = tf.expand_dims(self.w_z, axis=-1) # (N_samples, 1)-Tensor

        Zcpreds = self.f_pred([self.zc, self.zu]) # (B: N_samples, E: d_zc)-Dist

    def top_down(self):

        Zwtargets_comb = [[neighbor.w_z, f_trans.foreward(neighbor.Zcpreds)]
                          for neighbor, f_trans in self.neighbors.items()] + \
                         self.child_targets + [self.w_z, self.Zcpreds]
        w_Ztargets = tf.stack(Zwtargets_comb[:, 0], axis=0)
        # (N_targets x node.N_samples,)-tensor
        Ztargets = tf.stack(Zwtargets_comb[:, 1], axis=0)
        # (B: N_targets x node.N_samples, E:self.d_zc)-Dist
        self.Ztarget = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=w_Ztargets),
            components_distribution=Ztargets
        ) # (B: E:d_zc,)-dist

        ztargets = self.Ztarget.sample(N_SAMPLES) # (N_samples, self.d_zc)-Tensor
        Xparentstargets = self.f_act([self.zc, self.zu, ztargets])
        # (N_parents)-list< (B: N_samples, E: parent.d_zc)-Dist >

        w_Xtargets = tf.reduce_mean(self.w_z) - self.Ztarget.log_prob(ztargets) # (N_samples,) Tensor
        for parent, Xparenttarget in zip(self.parents, Xparentstargets):
            parent.child_targets.append([w_Xtargets, Xparenttarget])

    def train(self):
        # min KL from target not pred
        # train pred on actual data
        # train actor on inverse from pred
        pass







