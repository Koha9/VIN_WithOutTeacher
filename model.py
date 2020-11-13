import numpy as np

import tensorflow as tf


def attention(tensor, params):
    """Attention model for grid world domain
    """

    S1, S2 = params
    # Flatten
    #s1 = tf.reshape(S1, [-1])  #No USE??????
    #s2 = tf.reshape(S2, [-1])  #No USE??????
    s1 = tf.dtypes.cast(S1,tf.int32) #to int32
    s2 = tf.dtypes.cast(S2,tf.int32)

    # Indices for slicing
    N = tf.shape(input=tensor)[0] # [num] ([1,2,3,4,5,6,7 .....])
    idx = tf.stack([tf.range(N), s1, s2], axis=1) # [num,3]
    # Slicing values
    q_out = tf.gather_nd(tensor, idx, name='q_out') # [num,10] 按顺序提取位于s1，s2坐标的q值[10]

    return q_out


class VIN(tf.keras.Model):
    def __init__(self, chi=2, chh=150, chq=10):
        super().__init__()

        ch_i = chi  # Channels in input layer
        ch_h = chh  # Channels in initial hidden layer
        ch_q = chq  # Channels in q layer (~actions)
        self.conv0 = tf.keras.layers.Conv2D(filters=ch_h,
                                            kernel_size=[3, 3],
                                            strides=[1, 1],
                                            padding='same',
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer=tf.random_normal_initializer(
                                                stddev=0.01),
                                            bias_initializer=tf.zeros_initializer(),
                                            name='h0')
        self.conv1 = tf.keras.layers.Conv2D(filters=1,
                                            kernel_size=[3, 3],
                                            strides=[1, 1],
                                            padding='same',
                                            activation=None,
                                            use_bias=False,
                                            kernel_initializer=tf.random_normal_initializer(
                                                stddev=0.01),
                                            bias_initializer=None,
                                            name='r')
        self.conv2 = tf.keras.layers.Conv2D(filters=ch_q,
                                            kernel_size=[3, 3],
                                            strides=[1, 1],
                                            padding='same',
                                            activation=None,
                                            use_bias=False,
                                            kernel_initializer=tf.random_normal_initializer(
                                                stddev=0.01),
                                            bias_initializer=None,
                                            name='q')  # 初始化数据集
        self.conv_ShareWeights = tf.keras.layers.Conv2D(filters=ch_q,
                                                        kernel_size=[3, 3],
                                                        strides=[1, 1],
                                                        padding='same',
                                                        activation=None,
                                                        use_bias=False,
                                                        kernel_initializer=tf.random_normal_initializer(
                                                            stddev=0.01),
                                                        bias_initializer=None,
                                                        name='q')  # sharing weights
        self.conv_ShareWeights2 = tf.keras.layers.Conv2D(filters=ch_q,
                                                kernel_size=[3, 3],
                                                strides=[1, 1],
                                                padding='same',
                                                activation=None,
                                                use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer(
                                                    stddev=0.01),
                                                bias_initializer=None,
                                                name='q')  # sharing weights
        self.dense0 = tf.keras.layers.Dense(units=8,
                                            activation=None,
                                            use_bias=False,
                                            kernel_initializer=tf.random_normal_initializer(
                                                stddev=0.01),
                                            name='logits')

    @tf.function
    def call(self, inputs, S1, S2, VInum=10):
        k = VInum  # Number of Value Iteration computations

        h = self.conv0(inputs) # [num, img, img, 150]
        
        r = self.conv1(h) # [num, img, img, 1]

        v = tf.zeros_like(r)  # 初始化全0Value map 
        rv = tf.concat([r, v], axis=3) # [num, img, img, 2]

        q = self.conv2(rv)#[num, img,img,10]
        
        v = tf.reduce_max(q, axis=3, keepdims=True, name='v')# [num, img, img,1]
        
        for i in range(0, k-1): # value iteration
            rv = tf.concat([r, v], axis=3) # [num, img, img, 1+1]
            q = self.conv_ShareWeights(rv)  # [num, img, img, 10]
            v = tf.reduce_max(q, axis=3, keepdims=True, name='v') # [num, img,img,1]

        rv = tf.concat([r, v], axis=3) # [num, img, img, 2]
        q = self.conv_ShareWeights2(rv) # [num, img, img, 10]
        q_out = attention(tensor=q, params=[S1, S2]) # [num, 10]

        logits = self.dense0(q_out) # [num, 8] 每个动作的权重？

        prob_actions = tf.nn.softmax(logits, name='probability_actions') # [num, 8] 每个动作可能性，取值0~1

        return logits, prob_actions, q, q_out

