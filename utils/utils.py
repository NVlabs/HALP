# --------------------------------------------------------
# Copyright (C) 2022 NVIDIA Corporation. All rights reserved.
# 
# Official PyTorch implementation of NeurIPS2022 paper
# Structural Pruning via Latency-Saliency Knapsack
# Maying Shen, Hongxu Yin, Pavlo Molchanov, Lei Mao, Jianna Liu and Jose M. Alvarez
#
# This work is licensed under the NVIDIA Source Code License
# To view a copy of this license, see the LICENSE file.
# --------------------------------------------------------

import logging
import os
import sys

import torch.distributed as dist
import torch.nn as nn
import yaml


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    # stdout logging: master only
    if int(os.environ["RANK"]) == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(plain_formatter)
        logger.addHandler(ch)

    # file logging: all workers
    filename = os.path.join(os.environ["OUTPUT_PATH"], "train.log")
    if int(os.environ["RANK"]) > 0:
        filename = filename + ".rank{}".format(os.environ["RANK"])

    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(plain_formatter)
    logger.addHandler(fh)

    return logger


class ExpConfig():
    def __init__(self, cfg):
        for k, v in cfg.items():
            self._set_attribute(k, v)

    def _set_attribute(self, attr_name, attr_value):
        setattr(self, attr_name, attr_value)

    def override_config(self, source):
        for k, v in source.items():
            self._set_attribute(k, v)

    def save_config(self):
        attr_dict = self.__dict__
        with open(os.path.join(os.environ["OUTPUT_PATH"], "exp_config.yaml"), "w") as f:
            yaml.dump(attr_dict, f, default_flow_style=False)


def is_conv(layer):
    conv_type = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
    isConv = False
    for ctp in conv_type:
        if isinstance(layer, ctp):
            isConv = True
            break
    return isConv


def is_bn(layer):
    bn_type = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
    isbn = False
    for ctp in bn_type:
        if isinstance(layer, ctp):
            isbn = True
            break
    return isbn


def extract_layers(model, layers={}, pre_name='', get_conv=True, get_bn=False):
    """
        Get all the model layers and the names of the layers
    Returns:
        layers: dict, the key is layer name and the value is the corresponding model layer
    """
    for name, layer in model.named_children():
        new_name = '.'.join([pre_name,name]) if pre_name != '' else name
        if len(list(layer.named_children())) > 0:
            extract_layers(layer, layers, new_name, get_conv, get_bn)
        else:
            get_layer = False
            if get_conv and is_conv(layer):
                get_layer = True
            elif get_bn and is_bn(layer):
                get_layer = True
            if get_layer:
                layers[new_name] = layer
    return layers


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size() if dist.is_initialized() else 1
    return rt


def calc_ips(batch_size, time):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    tbs = world_size * batch_size
    return tbs/time
