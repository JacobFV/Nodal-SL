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


class LCA(tfkl.Layer):

    def __init__(self,
                 N_w,
                 lr_w=1e-3,
                 winner_take_all=True,
                 pool_act_axes=[],
                 backpropagatable=False,
                 minval=0, maxval=1):
        super(LCA, self).__init__()
        self.N_w = N_w
        self.lr_w = lr_w
        self.winner_take_all = winner_take_all
        self.pool_act_axes = pool_act_axes
        self.backpropagatable = backpropagatable
        self.minval = minval
        self.maxval = maxval

    def build(self, input_shape):
        d_x = input_shape[-1] # (1,)-Tensor

        self.ws = tf.random.uniform(shape=(d_x, self.N_w),
                                    minval=self.minval,
                                    maxval=self.maxval) # ws: (d_x, N_w)
        if self.backpropagatable:
            self.ws = tf.Variable(self.ws, trainable=True)


        # if not self.winner_take_all:
        #     self.batch_norm_layer = tfkl.BatchNormalization(axis=-2)

    def call(self, inputs, **kwargs):

        # inputs: (..., d_x)
        X = tf.expand_dims(inputs, axis=-1) # (..., d_x, 1)
        delta = X - self.ws # (..., d_x, N_w)
        dist = tf.norm(delta, ord=1, axis=-2) # (..., N_w)
        act = tf.exp(-dist) # (..., N_w)

        if self.winner_take_all:
            ind_ws = K.argmax(act, axis=-1) # (..., 1)
            act = tf.one_hot(ind_ws, depth=self.N_w, axis=-1) # (..., N_w)
        # else:
        #     act = self.batch_norm_layer(act) # (..., N_w)

        if 'training' in kwargs and kwargs['training']:
            # move weights closer to inputs
            dws = tf.sign(delta) * act[..., tf.newaxis, :] # (..., d_x, N_w)
            sum_axes = tf.range(tf.rank(dws)-3) # (?,)
            # eg: if `dws.shape=(B, N_x, d_x, N_w)`, then `rank(dws)=4`,
            # `sum_axes=tf.range(1)=[0,1]`. take sum over batch and input x axes.
            dws = tf.reduce_sum(dws, axis=sum_axes) # (d_x, N_w)
            self.ws = self.ws + self.lr_w * dws # (d_x, N_w)

        if len(self.pool_act_axes) > 0:
            act = tf.reduce_sum(act, axis=self.pool_act_axes)
            # eg: for an image, `X.shape=(B, N_y, N_x, N_c)`, `(B, N_y, N_x)` become the 'batch' axes
            # while the channels form the weight space. As a result, `act.shape=(B, N_y, N_x, N_w)`.
            # - if `self.pool_act_axes=[]`, this performs 1D convolution
            # - if `self.pool_act_axes=[-3,-2]`, this performs global feature pooling

        return act # (..., N_w) or some dimensional reduction therefrom


class SplitLCA(tfkl.Layer):

    def __init__(self, LCA_params, N_splits=2, **kwargs):
        super(SplitLCA, self).__init__(**kwargs)

        self.N_splits = N_splits
        self.LCA_params = LCA_params

        self.split_layers = list()
        self.LCA_layers = list()
        self.add_layer = tfkl.Add()

    def build(self, input_shape):

        split_size = input_shape[-1] // self.N_splits
        split_index = 0
        for i in range(self.N_splits):
            if i < self.N_splits-1:
                next_split_index = split_index + split_size
                self.split_layers.append(tfkl.Lambda(
                    lambda x: x[...,split_index:next_split_index]))
                split_index = next_split_index
            else: # i == self.N_splits-1:
                self.split_layers.append(tfkl.Lambda(
                    lambda x: x[...,split_index:]))
            self.LCA_layers.append(LCA(**self.LCA_params))

    def call(self, inputs, **kwargs):
        return self.add_layer([LCA_layer(split_layer(inputs), **kwargs)
                               for LCA_layer, split_layer
                               in zip(self.LCA_layers, self.split_layers)])


class SemisupervisedLCA(tfkl.Layer):

    def __init__(self,
                 d_supervised, d_unsupervised,
                 N_splits,
                 LCA_params=None, dense_params=None,
                 **kwargs):
        super(SemisupervisedLCA, self).__init__(**kwargs)

        if LCA_params is None:
            LCA_params = dict()
        LCA_params.update(dict(d_w=d_unsupervised))
        if dense_params is None:
            dense_params = dict(activation='relu')
        dense_params.update(dict(units=d_supervised))

        self.LCA_layer = \
            SplitLCA(LCA_params=LCA_params, N_splits=N_splits) \
                if N_splits != 1 else LCA(**LCA_params)
        self.dense_layer = tfkl.Dense(**dense_params)
        self.concat_layer = tfkl.Concatenate()

    def call(self, inputs, **kwargs):
        dense_features = self.dense_layer(inputs)
        splitLCA_features = self.LCA_layer(inputs)
        return self.concat_layer([dense_features, splitLCA_features])


class Conv2DLCA(LCA):

    def __init__(self,
                 filters,
                 kernel_size=(3,3),
                 strides=1,
                 padding='valid',
                 lr_w=1e-3,
                 winner_take_all=True,
                 backpropagatable=False):

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert isinstance(kernel_size, tuple) and len(kernel_size) == 2

        if isinstance(strides, int):
            strides = (strides, strides)
        assert isinstance(strides, tuple) and len(strides) == 2

        padding = padding.lower().strip()
        assert padding == 'valid' or padding == 'same'

        super(Conv2DLCA, self).__init__(N_w=filters,
                                        lr_w=lr_w,
                                        winner_take_all=winner_take_all,
                                        pool_act_axes=[],
                                        backpropagatable=backpropagatable)

        self.kernel_size = tf.constant(kernel_size)
        self.strides = tf.constant(strides)
        self.padding = padding

    def build(self, input_shape):
        example_ones = tf.ones(input_shape)
        example_reshaped = self._preprocess_image(example_ones)
        super(Conv2DLCA, self).build(tf.shape(example_reshaped))

    def call(self, inputs, **kwargs):
        cropped_image = self._preprocess_image(inputs)
        return super(Conv2DLCA, self).call(cropped_image, **kwargs)

    def _preprocess_image(self, inputs):
        tf.assert_rank(inputs, 4) # [B, Y, X, C]
        # you must add an empty dimension (`[..., None]`)
        # to feed greyscale images to this Conv2D layer
        kernel_cutoff = (self.kernel_size-1)//2

        # pad so opposite edges don't roll over to each other
        padded_image = tf.pad(inputs, [(0, 0),
                                       (kernel_cutoff[0], kernel_cutoff[0]),
                                       (kernel_cutoff[1], kernel_cutoff[1]),
                                       (0, 0)])

        # stagger stack axis 1
        y_staggered_stack = tf.concat([
            tf.roll(input=padded_image, shift=N, axis=1)
            for N in range(-kernel_cutoff[0], kernel_cutoff[0]+1)
        ], axis=-1)

        # stagger stack axis 2
        xy_staggered_stack = tf.concat([
            tf.roll(input=y_staggered_stack, shift=N, axis=2)
            for N in range(-kernel_cutoff[1], kernel_cutoff[1]+1)
        ], axis=-1)

        # cutoff padding
        if self.padding == 'same':
            cropped_image = xy_staggered_stack[:,
                                               kernel_cutoff[0]:-kernel_cutoff[0],
                                               kernel_cutoff[1]:-kernel_cutoff[1],
                                               :]
        elif self.padding == 'valid':
            cropped_image = xy_staggered_stack[:,
                                               2*kernel_cutoff[0]:-2*kernel_cutoff[0],
                                               2*kernel_cutoff[1]:-2*kernel_cutoff[1],
                                               :]
        else:
            raise NotImplementedError()
        return cropped_image


class DepthwiseConv2DLCA(tfkl.Layer):

    def __init__(self, name=None, **kwargs):
        super(DepthwiseConv2DLCA, self).__init__(name=name)
        self.kwargs = kwargs

    def build(self, input_shape):
        assert input_shape.rank == 4
        N_channels = input_shape[-1]
        self.channel_convs = [Conv2DLCA(**self.kwargs) for _ in range(N_channels)]
        self.final_conv = Conv2DLCA(**self.kwargs)

    def call(self, inputs, **kwargs):
        tf.assert_rank(inputs, 4) # [B, Y, X, C]
        channel_images = tf.unstack(inputs, axis=-1)
        channel_activations = [channel_conv(channel_image[..., tf.newaxis], **kwargs)
                               for channel_conv, channel_image
                               in zip(self.channel_convs, channel_images)]
        channel_activations = tf.concat(channel_activations, axis=-1)
        final_activations = self.final_conv(channel_activations, **kwargs)
        return final_activations

