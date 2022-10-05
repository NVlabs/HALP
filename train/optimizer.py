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

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example."
    )
import numpy as np
import torch

from utils.utils import get_logger


logger = get_logger("train.optimizer")

def get_optimizer(model, exp_cfg, resume_optimizer_state_dict=None):
    parameters = list(model.named_parameters())
    logger.info(" ! Weight decay applied to parameters: {}".format(exp_cfg.weight_decay))
    logger.info(" ! Weight decay applied to BN parameters: {}".format(exp_cfg.bn_weight_decay))
    total_size_params = sum([np.prod(par[1].shape) for par in parameters])
    logger.info("Total number of trainable parameters: {}".format(total_size_params))
    bn_params = [v for n, v in parameters if "bn" in n]
    rest_params = [v for n, v in parameters if not "bn" in n]
    optimizer = torch.optim.SGD(
        [
            {"params": bn_params, "weight_decay": exp_cfg.bn_weight_decay},
            {"params": rest_params, "weight_decay": exp_cfg.weight_decay}
        ],
        exp_cfg.learning_rate,
        momentum=exp_cfg.momentum,
        weight_decay=exp_cfg.weight_decay,
        nesterov=exp_cfg.nesterov
    )
    if exp_cfg.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=exp_cfg.static_loss_scale,
                                   dynamic_loss_scale=exp_cfg.dynamic_loss_scale,
                                   verbose=False)
    if resume_optimizer_state_dict is not None:
        optimizer.load_state_dict({k.replace("module.", ""): v for k, v in resume_optimizer_state_dict.items()})
        logger.info("==> Loaded the optimizer state dict.")
    
    if exp_cfg.amp:
        amp.register_float_function(torch, "batch_norm")
        model, optimizer = amp.initialize(
            model, optimizer,
            opt_level=exp_cfg.opt_level,
            loss_scale="dynamic" if exp_cfg.dynamic_loss_scale else exp_cfg.static_loss_scale
        )

    return model, optimizer
