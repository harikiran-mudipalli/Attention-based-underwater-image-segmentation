## https://arxiv.org/pdf/2010.03045.pdf
## https://github.com/jyhengcoder/Triplet-Attention-tf/blob/main/triplet_attention.py

import tensorflow as tf
from tensorflow.keras import layers
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import BatchNormalization, Activation, MaxPooling2D
#from utils import conv_kernel_initializer

class BasicConv(object):
    def __init__(self, out_planes, kernel_size):
        super(BasicConv, self).__init__()
        self.conv = Conv2D(
            out_planes,
            kernel_size=[kernel_size, kernel_size],
            strides=[1, 1],
            padding='same',
            kernel_initializer='glorot_uniform',
            use_bias=False,
            data_format='channels_first')
        self.bn = BatchNormalization(
                axis=-1,
                momentum=0.999,
                epsilon=1e-5,
                fused=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = tf.nn.relu(x)
        return x

class ChannelPool(object):
    def forward(self, x):
        return tf.concat( [tf.expand_dims(tf.reduce_max(x, axis=1), axis=1),
                tf.expand_dims(tf.reduce_mean(x, axis=1), axis=1)], axis=1 )

class SpatialGate(object):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = BasicConv(1, kernel_size)
    def forward(self, x):
        x_compress = self.compress.forward(x)
        x_out = self.spatial.forward(x_compress)
        scale = tf.nn.sigmoid(x_out)
        return x * scale

class TripletAttention(object):
    def __init__(self, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        print("Triplet Attention!")
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = tf.transpose(x, perm=[0, 2, 1, 3])
        x_out1 = self.ChannelGateH.forward(x_perm1)
        x_out11 = tf.transpose(x_out1, perm=[0, 2, 1, 3])
        x_perm2 = tf.transpose(x, perm=[0, 3, 2, 1])
        x_out2 = self.ChannelGateW.forward(x_perm2)
        x_out21 = tf.transpose(x_out2, perm=[0, 3, 2, 1])
        if not self.no_spatial:
            x_out = self.SpatialGate.forward(x)
            x_out = (1/3)*(x_out + x_out11 + x_out21)
        else:
            x_out = (1/2)*(x_out11 + x_out21)
        return x_out
