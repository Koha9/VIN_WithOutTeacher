import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import Model


class VIN(Model):
    def __init__(self, k=50, ch_h=150, ch_q=10, imsize=28):
        super(VIN, self).__init__()

        self.k = k
        self.ch_h = ch_h  # Channels in initial hidden layer
        self.ch_q = ch_q  # Channels in q layer (~actions)
        self.imsize = imsize

        # CNN阶段
        self.h1 = Conv2D(filters=self.ch_h,
                         kernel_size=[3, 3],
                         strides=[1, 1],
                         padding='same',
                         activation=None,
                         use_bias=True,
                         kernel_initializer=tf.random_normal_initializer(
                             stddev=0.01),
                         bias_initializer='zeros',
                         name='h1',)

        self.r1 = Conv2D(filters=1,
                         kernel_size=[3, 3],
                         strides=[1, 1],
                         padding='same',
                         activation=None,
                         use_bias=False,
                         kernel_initializer=tf.random_normal_initializer(
                             stddev=0.01),
                         bias_initializer=None,
                         name='r1',)

        # VIN阶段
        self.q1 = Conv2D(filters=self.ch_q,
                         kernel_size=[3, 3],
                         strides=[1, 1],
                         padding='same',
                         activation=None,
                         use_bias=False,
                         kernel_initializer=tf.random_normal_initializer(
                             stddev=0.01),
                         bias_initializer=None,
                         name='q1',)
        # 全连接层
        self.l1 = tf.keras.layers.Dense(units=8,
                                        activation=None,
                                        use_bias=False,
                                        kernel_initializer=tf.random_normal_initializer(
                                            stddev=0.01),
                                        name='logits')

    def call(self, total):  # Number of Value Iteration computations
        S1 = tf.cast(total[:, 0], tf.int32)
        S2 = tf.cast(total[:, 1], tf.int32)

        O = total[:, 2:]
        X = tf.reshape(O, (-1, self.imsize, self.imsize, 2))
        #X = tf.transpose(O, perm=[0, 3, 1, 2])
        # CNN部分
        h = self.h1(X)
        r = self.r1(h)
        # 保存报酬MAP

        # VIN部分
        v = tf.zeros_like(r)
        rv = tf.concat([r, v], 3)
        q = self.q1(rv)
        v = tf.reduce_max(q, axis=3, keepdims=True)
        for i in range(0, self.k-1):
            rv = tf.concat([r, v], 3)
            q = self.q1(rv)
            v = tf.reduce_max(q, axis=3, keepdims=True)
        # 保存价值MAP

        # 最后一次卷积
        rv = tf.concat([r, v], 3)
        q = self.q1(rv)
        v = tf.reduce_max(q, axis=3, keepdims=True)
        # 进入注意力模块
        q_out = self.attention(tensor=q, parmas=[S1, S2])

        logits = self.l1(q_out)
        prob_actions = tf.nn.softmax(logits, name='probability_actions')
        return v  # , q

    def attention(self, tensor, parmas):
        S1, S2 = parmas

        # 拉直
        s1 = tf.reshape(S1, [-1])
        s2 = tf.reshape(S2, [-1])

        N = tf.shape(tensor)[0]  # 地图数
        # S1S2和标签list0-150？在第一维度结合，[0,2,5][1,8,6][2,7,6]的感觉？
        idx = tf.stack([tf.range(N), s1, s2], axis=1)

        # 切片
        # 貌似返回的是每一张图坐标对应位置的数值 ????????????
        q_out = tf.gather_nd(tensor, idx, name='q_out')

        return q_out
