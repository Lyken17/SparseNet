#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import os

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *


def linearFetch(lst):
    return lst


def expFetch(lst):
    nums = len(lst)
    i = 1
    res = []
    while i <= nums:
        res.append(lst[nums - i])
        i *= 2
    return res


def conv3x3(name, input_data, out_channels, stride=1):
    return Conv2D(name, input_data, out_channels, 3, stride=stride, nl=tf.identity, use_bias=False,
                  W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / out_channels)))


def conv1x1(name, input_data, out_channels, stride=1):
    return Conv2D(name, input_data, out_channels, 1, stride=stride, nl=tf.identity, use_bias=False,
                  W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / out_channels)))


def add_layer_without_concat(name, l, growth_rate):
    # basic BN-ReLU-Conv unit
    with tf.variable_scope(name) as scope:
        c = BatchNorm('bn1', l)
        c = tf.nn.relu(c)
        c = conv3x3('conv1', c, growth_rate, 1)
    return c


def add_layer(name, l, growth_rate):
    c = add_layer_without_concat(name, l, growth_rate)
    return tf.concat([c, l], 3)


def add_bottleneck_without_concat(name, l, growth_rate, dropout=0):
    inter_channels = 4 * growth_rate
    # complex BN-ReLU-Conv-BN-ReLU-Conv
    with tf.variable_scope(name) as scope:
        c = BatchNorm('bn1', l)
        c = tf.nn.relu(c, 'relu1')
        c = conv1x1('conv1', c, out_channels=inter_channels)
        if dropout > 0:
            c = tf.nn.dropout(c, dropout)

        c = BatchNorm('bn2', l)
        c = tf.nn.relu(c, 'relu2')
        c = conv1x1('conv2', c, out_channels=growth_rate)
        if dropout > 0:
            c = tf.nn.dropout(c, dropout)
    return c


def add_bottleneck(name, l, growth_rate, dropout):
    c = add_bottleneck_without_concat(name, l, growth_rate, dropout)
    return tf.concat([c, l], 3)


def add_transition(name, l):
    shape = l.get_shape().as_list()
    in_channel = shape[3]
    with tf.variable_scope(name) as scope:
        l = BatchNorm('bn1', l)
        l = tf.nn.relu(l)
        l = Conv2D('conv1', l, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu)
        l = AvgPooling('pool', l, 2)
    return l
