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

from argparse import ArgumentParser, REMAINDER
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    """Parse the command line arguments."""

    parser = ArgumentParser(
        description=(
            "PyTorch distributed training launch helper utilty that will spawn up "
            "multiple distributed processes."
        ),
    )

    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="The number of nodes to use for distributed training.",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="The rank of the node for distributed training.",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=8,
        help=(
            "The number of processes to launch on each node"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory to save the checkpoint and training log.",
    )

    parser.add_argument(
        "main_script",
        type=str,
        help=(
            "The path to the single GPU training program/script to be launched in parallel, "
            "followed by all the arguments for the training script"
        )
    )
    # rest from the training program
    parser.add_argument(
        "main_script_args",
        nargs=REMAINDER
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.nnodes

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = "127.0.0.1"
    current_env["MASTER_PORT"] = "29500"
    current_env["WORLD_SIZE"] = str(dist_world_size)

    processes = []

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    current_env["OUTPUT_PATH"] = args.output_dir

    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)

        # spawn the processes
        cmd = [
            sys.executable,
            "-u",
            args.main_script,
            "--local_rank={}".format(local_rank)
        ] + args.main_script_args

        print(cmd)

        process = subprocess.Popen(cmd, env=current_env, stdout=None)
        processes.append(process)

    try:
        up = True
        error = False
        while up and not error:
            up = False
            for p in processes:
                ret = p.poll()
                if ret is None:
                    up = True
                elif ret != 0:
                    error = True
            time.sleep(1)

        if error:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            exit(1)

    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
        raise
    except SystemExit:
        for p in processes:
            p.terminate()
        raise
    except:
        for p in processes:
            p.terminate()
        raise


if __name__ == "__main__":
    main()
