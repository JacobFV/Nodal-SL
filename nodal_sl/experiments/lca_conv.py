from ..core.lca import Conv2DLCA, DepthwiseConv2DLCA

import tensorflow as tf
keras = tf.keras
tfkl = keras.layers

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
_, height, width, channels = x_train.shape
print('data shape:', x_train.shape)

base_layers = [
    DepthwiseConv2DLCA(filters=32),
    DepthwiseConv2DLCA(filters=128),
]

head_layers = [
    tfkl.Flatten(),
    tfkl.Dense(128, activation='relu'),
    tfkl.Dense(10, activation='softmax', use_bias=False)
]

model = keras.Sequential([
    tfkl.Input((height, width, channels)),
    base_layers,
    head_layers
])
model.compile(optimizer='sgd', loss='crossentropy', metrics='accuracy')
model.fit(x_train, y_train)