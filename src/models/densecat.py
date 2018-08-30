#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import os

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

"""
CIFAR10 DenseNet example. See: http://arxiv.org/abs/1608.06993
Code is developed based on Yuxin Wu's ResNet implementation: https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet
Results using DenseNet (L=40, K=12) on Cifar10 with data augmentation: ~5.77% test error.

Running time:
On one TITAN X GPU (CUDA 7.5 and cudnn 5.1), the code should run ~5iters/s on a batch size 64.

Reimplementation notes:

"""

BATCH_SIZE = 64


class Model(ModelDesc):
    def __init__(self, depth, growth_rate=12, fetch="dense",
                 bottleneck=False, compression=1, dropout=0,
                 num_classes=10):

        super(Model, self).__init__()
        self.N = int((depth - 4) / 3)
        self.growthRate = growth_rate
        self.fetch = fetch
        self.bottleneck = bottleneck
        self.compression = compression
        self.num_classes = num_classes


    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, input_vars):
        image, label = input_vars
        image = image / 128.0 - 1 # convert range to -1 ~ 1

        from .utils import conv3x3, conv1x1, add_layer, add_layer_without_concat, add_transition


        def dense_net(name, num_classes=10):
            l = conv3x3('conv0', image, 16, 1)

            from .utils import linearFetch, expFetch
            if self.fetch == "dense":
                fetch = linearFetch
            else:
                fetch = expFetch

            with tf.variable_scope('block1') as scope:
                saved_activations = [l]
                for i in range(self.N):
                    l = add_layer_without_concat('densen_layer.{}'.format(i), l, self.growthRate)
                    saved_activations.append(l)

                    l = tf.concat(fetch(saved_activations), 3)
                l = add_transition('transition1', l)

            with tf.variable_scope('block2') as scope:
                saved_activations = [l]
                for i in range(self.N):
                    l = add_layer_without_concat('densen_layer.{}'.format(i), l, self.growthRate)
                    saved_activations.append(l)
                    l = tf.concat(fetch(saved_activations), 3)
                l = add_transition('transition2', l)

            with tf.variable_scope('block3') as scope:
                saved_activations = [l]
                for i in range(self.N):
                    l = add_layer_without_concat('densen_layer.{}'.format(i), l, self.growthRate)
                    saved_activations.append(l)
                    l = tf.concat(fetch(saved_activations), 3)

            l = BatchNorm('bn_last', l)
            l = tf.nn.relu(l)
            l = GlobalAvgPooling('gap', l)
            logits = FullyConnected('linear', l, out_dim=self.num_classes, nl=tf.identity)

            return logits

        logits = dense_net("dense_net")

        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label)

        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W
        wd_cost = tf.multiply(1e-4, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))  # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
