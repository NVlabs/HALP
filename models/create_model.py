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

from models.resnet import make_resnet
from models.resnet_pruned import make_pruned_resnet


def get_model(arch, class_num, enable_bias, group_mask_file=None):
    if group_mask_file is None:
        if "resnet" in arch.lower():
            model = make_resnet(arch, class_num, enable_bias)
        else:
            raise NotImplementedError
    else:
        if "resnet" in arch.lower():
            model = make_pruned_resnet(arch, group_mask_file, class_num, enable_bias)
        else:
            raise NotImplementedError

    return model
