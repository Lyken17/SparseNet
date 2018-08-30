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
"""

BATCH_SIZE = 64


def get_data(train_or_test, args):
    isTrain = train_or_test == 'train'
    if args.dataset == "c10":
        dst = dataset.Cifar10
    elif args.dataset == "c100":
        dst = dataset.Cifar100

    ds = dst(train_or_test)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [imgaug.CenterPaste((40, 40)), imgaug.RandomCrop((32, 32)), imgaug.Flip(horiz=True),
                      # imgaug.Brightness(20),
                      # imgaug.Contrast((0.6,1.4)),
                      imgaug.MapImage(lambda x: x - pp_mean), ]
    else:
        augmentors = [imgaug.MapImage(lambda x: x - pp_mean)]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds


def get_config(args):
    log_dir = 'train_log/%s-%d-%d-%s-single-fisrt%s-second%s-max%s' % (
        str(args.fetch), args.depth, args.growth_rate,
        args.dataset,
        str(args.drop_1), str(args.drop_2), str(args.max_epoch))
    logger.set_logger_dir(log_dir, action='n')

    # prepare dataset
    dataset_train = get_data('train', args)
    steps_per_epoch = dataset_train.size()
    dataset_test = get_data('test', args)

    from models.densecat import Model

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[ModelSaver(), InferenceRunner(dataset_test, [ScalarStats('cost'), ClassificationError()]),
                   ScheduledHyperParamSetter('learning_rate', [(1, 0.1), (args.drop_1, 0.01), (args.drop_2, 0.001)])],
        model=Model(
            depth=args.depth, growth_rate=args.growth_rate, fetch=args.fetch, num_classes=args.num_classes),
        steps_per_epoch=steps_per_epoch,
        max_epoch=args.max_epoch,
    )


if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument('-g', '--gpu', help='comma separated list of GPU(s) to use.')  # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    parser.add_argument('--log-dir', default="train_log", help="The root directory to save training logs.")
    parser.add_argument('--dataset', default="c10", type=str, choices=["c10", "c100"])
    parser.add_argument('--name', default=None)

    # model related
    # parser.add_argument('--arch')
    parser.add_argument('--fetch', default="dense", type=str, choices=["dense", "sparse"])
    parser.add_argument('-d', '--depth', default=40, type=int, help='The depth of densenet')
    parser.add_argument('-gr','--growth-rate', default=12, type=int, help='The number of output filters ')
    parser.add_argument('--growth-step', default=None)

    parser.add_argument('--bottleneck', default=0, type=int, help="Whether to use bottleneck")
    parser.add_argument('--compression', default=0, type=float, help="Whether to use compression")
    parser.add_argument('--dropout', default=0.0, type=float, help="The ratio of dropout layer")

    # optimizer
    parser.add_argument('--batch-size', default=64, type=int, help="Batch fed into graph every iter")
    parser.add_argument('--drop_1', default=150, help='Epoch to drop learning rate to 0.01.')  # nargs='*' in multi mode
    parser.add_argument('--drop_2', default=225, help='Epoch to drop learning rate to 0.001')
    parser.add_argument('--max_epoch', default=300, help='max epoch')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    BATCH_SIZE = args.batch_size

    nr_tower = 0
    if args.gpu:
        nr_tower = len(args.gpu.split(','))
        BATCH_SIZE = BATCH_SIZE // nr_tower

    if args.dataset == 'c10':
        args.num_classes = 10
    elif args.dataset == 'c100':
        args.num_classes = 100
    else:
        raise NotImplementedError

    config = get_config(args)
    if args.load:
        config.session_init = SaverRestore(args.load)

    # SyncMultiGPUTrainer(config).train()
    launch_train_with_config(config, SyncMultiGPUTrainer(nr_tower))
