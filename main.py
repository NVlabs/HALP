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

import argparse
import os
import random

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import yaml

from data.dataloaders import get_train_loader, get_val_loader
from models import get_model
from prune.pruner import Pruner
from train.optimizer import get_optimizer
from train.lr_schedule import lr_cosine_policy, lr_step_policy
from train.training import train_loop, validate
from utils.mixup import MixUpWrapper, NLLMultiLabelSmooth
from utils.smoothing import LabelSmoothing
from utils.utils import ExpConfig, get_logger


parser = argparse.ArgumentParser(
    description="Main script for running HALP algorithm."
)
parser.add_argument(
    "--exp", type=str, default="configs/exp_configs/rn50_imagenet_baseline.yaml",
    help="Config file for the experiment."
)
parser.add_argument(
    "--local_rank", type=int, default=0
)
parser.add_argument(
    "--seed", type=int, default=None,
    help="Random seed used for np and pytorch."
)
parser.add_argument(
    "--pretrained", type=str, default=None,
    help="The path of the pretrained model."
)
parser.add_argument(
    "--eval_only", action="store_true"
)
parser.add_argument(
    "--no_prune", action="store_true"
)
parser.add_argument(
    "--mask", type=str, default=None,
    help="The path of the mask file."
)
args = parser.parse_args()


def main():
    with open(args.exp) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    exp_cfg = ExpConfig(cfg)
    exp_cfg.override_config(vars(args))
    exp_cfg.save_config()

    ##### Environment Setup #####
    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    logger = get_logger("main")

    gpu = 0
    world_size = 1
    cuda = True
    if distributed:
        gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(gpu)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        world_size = torch.distributed.get_world_size()

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed + args.local_rank)
        torch.cuda.manual_seed(args.seed + args.local_rank)
        np.random.seed(seed=args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)
        def _worker_init_fn(id):
            np.random.seed(seed=args.seed + args.local_rank + id)
            random.seed(args.seed + args.local_rank + id)
    else:
        def _worker_init_fn(id):
            pass

    if exp_cfg.amp and exp_cfg.fp16:
        print("Please use only one of the --fp16/--amp flags")
        exit(1)  
    if exp_cfg.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
    if exp_cfg.static_loss_scale != 1.0:
        if not exp_cfg.fp16:
            logger.warning("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    ##### Loss #####
    loss = nn.CrossEntropyLoss
    if exp_cfg.mixup > 0.0:
        loss = lambda: NLLMultiLabelSmooth(exp_cfg.label_smoothing)
    elif exp_cfg.label_smoothing > 0.0:
        loss = lambda: LabelSmoothing(exp_cfg.label_smoothing)
    criterion = loss()
    if cuda:
        criterion = criterion.cuda()

    ##### Dataset #####
    train_loader, train_loader_len = get_train_loader(
        exp_cfg.data_root,
        exp_cfg.batch_size,
        exp_cfg.class_num,
        exp_cfg.mixup > 0.0,
        workers=exp_cfg.workers,
        fp16=(exp_cfg.fp16 or exp_cfg.amp),
        dataset=exp_cfg.dataset_name,
    )
    if exp_cfg.mixup != 0.0:
        train_loader = MixUpWrapper(exp_cfg.mixup, exp_cfg.class_num, train_loader)
    val_loader, val_loader_len = get_val_loader(
        exp_cfg.data_root,
        exp_cfg.batch_size,
        exp_cfg.class_num,
        workers=exp_cfg.workers,
        fp16=(exp_cfg.fp16 or exp_cfg.amp),
        dataset=exp_cfg.dataset_name,
    )

    ##### TensorBoard writer #####
    is_main = (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
    train_writer = None
    if is_main:
        train_writer = SummaryWriter(os.path.join(os.environ["OUTPUT_PATH"], "tf_event"))

    ##### Model #####
    model = get_model(
        exp_cfg.arch, exp_cfg.class_num, exp_cfg.enable_bias, group_mask_file=args.mask
    )
    logger.info(model)
    if args.pretrained is not None:
        resume_ckpt = torch.load(args.pretrained, map_location="cpu")
        if "state_dict" in resume_ckpt:
            resume_ckpt_state_dict = resume_ckpt["state_dict"]
        else:
            resume_ckpt_state_dict = resume_ckpt
        model.load_state_dict(
            {k.replace("module.", ""): v for k, v in resume_ckpt_state_dict.items()}
        )
        logger.info("Loaded model state_dict from {}".format(args.pretrained))
    if cuda:
        model = model.cuda()
    if exp_cfg.fp16:
        model = network_to_half(model)

    ##### Optimizer #####
    model, optimizer = get_optimizer(model, exp_cfg)

    ##### DDP #####
    if distributed:
        model = DDP(model, delay_allreduce=True)
    else:
        model = torch.nn.DataParallel(model)

    if exp_cfg.optimizer_batch_size < 0:
        batch_size_multiplier = 1
    else:
        tbs = world_size * exp_cfg.batch_size
        if exp_cfg.optimizer_batch_size % tbs != 0:
            logger.warning(
                "Warning: simulated batch size {} is not divisible by actual batch size {}".format(
                exp_cfg.optimizer_batch_size, tbs)
            )
        batch_size_multiplier = int(exp_cfg.optimizer_batch_size/tbs)
        logger.info("BSM: {}".format(batch_size_multiplier))

    ##### Eval only #####
    if args.eval_only:
        prec1, prec5 = validate(val_loader, model, criterion)
        logger.info("top1 accuracy: {}, top5 accuracy: {}".format(prec1, prec5))
        return

    ##### LR scheduler #####
    lr_scheduler = None
    if exp_cfg.lr_schedule == "step":
        lr_scheduler = lr_step_policy(exp_cfg.learning_rate, [30,60,80], 0.1, exp_cfg.warmup)
    elif exp_cfg.lr_schedule == "cosine":
        lr_scheduler = lr_cosine_policy(exp_cfg.learning_rate, exp_cfg.warmup, exp_cfg.epochs)

    ##### Pruner #####
    if args.no_prune:
        pruner = None
    else:
        pruner = Pruner(model, exp_cfg)

    ##### Train #####
    best_prec1, best_prec5 = train_loop(
        exp_cfg,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        batch_size_multiplier=batch_size_multiplier,
        train_writer=train_writer,
        pruner=pruner,
    )
    logger.info("Experiment ended")
    logger.info("Best top1 accuracy: {}, top5 accuracy: {}".format(best_prec1, best_prec5))

    pruner.print_neuron_num()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()
