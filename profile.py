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

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml

from models import get_model
from utils.model_summary import model_summary
from utils.utils import ExpConfig


parser = argparse.ArgumentParser(
    description="Main script for running HALP algorithm."
)
parser.add_argument(
    "--exp", type=str, default="configs/exp_configs/rn50_imagenet_baseline.yaml",
    help="Config file for the experiment."
)
parser.add_argument(
    "--model_path", type=str, default=None,
    help="The path of the model."
)
parser.add_argument(
    "--mask_path", type=str, required=True,
    help="The path of the mask file."
)
parser.add_argument(
    "--batch_size", type=int, default=256,
    help="The batch size of inference."
)
args = parser.parse_args()

def main():
    with open(args.exp) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    exp_cfg = ExpConfig(cfg)
    exp_cfg.override_config(vars(args))

    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.set_grad_enabled(False)
    gpu = 0
    cuda = True

    model = get_model(
        exp_cfg.arch, exp_cfg.class_num, exp_cfg.enable_bias, group_mask_file=args.mask_path
    )
    if args.model_path is not None:
        resume_ckpt = torch.load(args.model_path, map_location="cpu")
        if "state_dict" in resume_ckpt:
            resume_ckpt_state_dict = resume_ckpt["state_dict"]
        else:
            resume_ckpt_state_dict = resume_ckpt
        model.load_state_dict(
            {k.replace("module.", ""): v for k, v in resume_ckpt_state_dict.items()}
        )
    device = torch.device(gpu)

    model.eval()
    model.to(device)

    if exp_cfg.dataset_name.lower() == "imagenet":
        input = torch.randn(exp_cfg.batch_size, 3, 224, 224)
    elif exp_cfg.dataset_name.lower() == "cifar10":
        input = torch.randn(exp_cfg.batch_size, 3, 32, 32)
    else:
        raise NotImplementedError

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    times = []
    for i in range(40):
        input = input.to(device)
        start_evt.record()
        output = model(input)
        end_evt.record()
        torch.cuda.synchronize()
        elapsed_time = start_evt.elapsed_time(end_evt)
        # warmup
        if i < 10:
            continue
        times.append(elapsed_time)
    print("Infer time (ms/image)", np.mean(times) / exp_cfg.batch_size)
    print("FPS:", exp_cfg.batch_size * 1e+3 / np.mean(times))

    if exp_cfg.dataset_name.lower() == "imagenet":
        input = torch.randn(1, 3, 224, 224)
    elif exp_cfg.dataset_name.lower() == 'cifar10':
        input = torch.randn(1, 3, 32, 32)
    else:
        raise NotImplementedError
    flops = model_summary(model, input.cuda())
    print('MACs(G): {:.3f}'.format(flops / 1e9))


if __name__ == "__main__":
    main()
