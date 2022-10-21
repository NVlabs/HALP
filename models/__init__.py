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

from models.create_model import get_model

__all__ = (
    get_model
)
