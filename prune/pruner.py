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

import itertools
import json
import math
import os
import pickle as pkl

import numpy as np
import torch

from prune.cost import CostCalculator
from prune.importance import ChannelImportance
from prune.prune_config import PruneConfigReader
from utils.utils import extract_layers, get_logger


logger = get_logger("pruner")

class Pruner():
    def __init__(self, model, exp_cfg):
        self._prune_start = exp_cfg.prune_start_iter
        self._prune_interval = exp_cfg.prune_interval
        self._prune_steps = exp_cfg.prune_steps
        self._prune_end = self._prune_start + self._prune_interval * self._prune_steps
        self._disable_layer_prune = exp_cfg.disable_layer_prune

        self._group_mask = {}

        self.layers = extract_layers(model, get_conv=True, get_bn=True)
        config_reader = PruneConfigReader(exp_cfg.layer_cfg)
        self._groups = config_reader.prune_groups
        self._pre_group = config_reader.pre_group
        self._conv_bn = config_reader.conv_bn
        with open(exp_cfg.fmap_cfg, "r") as f:
            self._fmap_table = json.load(f)
        with open(exp_cfg.group_size_cfg, "r") as f:
            self._channel_group_size = json.load(f)

        self._importance_calculator = ChannelImportance()
        self._cost_calculator = CostCalculator(
            self.layers,
            self._groups,
            self._pre_group,
            self._fmap_table,
            exp_cfg.latency_lut_file,
            exp_cfg.lut_bs,
        )

        initial_latency = self._cost_calculator.get_total_latency(self._group_mask)
        self._prune_target = set_latency_prune_target(
            initial_latency,
            self._prune_start,
            self._prune_steps,
            exp_cfg.prune_ratio
        )

    
    def update_metric(self, global_step):
        if self._prune_start <= global_step < self._prune_end:
            self._importance_calculator.update_neuron_metric(
                self.layers, self._groups, self._conv_bn
            )

    
    def prune_step(self, global_step):
        pruned_num = 0
        if (
            self._prune_start <= global_step < self._prune_end and 
            (global_step + 1 - self._prune_start) % self._prune_interval == 0
        ):
            logger.info("Performing pruning.")
            self._importance_calculator.get_average_importance()
            pruned_num = self._latency_target_pruning(target_latency=self._prune_target.pop(0))
            self.save_mask(os.path.join(os.environ["OUTPUT_PATH"], "group_mask.pkl"))
            self._importance_calculator.reset_importance()
        else:
            self.mask_weights()
        
        return pruned_num

    
    def save_mask(self, path):
        """Save the pruning mask into a '.pkl' file.

        Args:
            path (str): The path for the file saving.
        """
        with open(path, "wb") as outfile:
            pkl.dump(
                {
                    group_name: mask.cpu()
                    for group_name, mask in self._group_mask.items()
                },
                outfile,
            )

    
    def _latency_target_pruning(self, target_latency):
        """Apply latency targeted pruning to the model.
        
        Apply latency targeted pruning to make the pruned network under the given
        latency constraint. Zero the pruned channel weights.

        Args:
            target_latency (float): The targeted latency after pruning.
        Returns:
            total_pruned_num (int): the total number of neurons being pruned.
        """
        int_scale = 1000
        # `grouped_neurons` saved as:
        # [{"layer_group_name": name, "channel_indices": [idx], "combined_importance": value}]
        # In each layer, neuron group with higher importance score has larger index.
        grouped_neurons = self._group_neurons_by_rank()
        # `adaptive_group_latency_change` saved as {group_name: [latency_change]}
        # In each layer, the list of latency change is calculated by decreasing the output
        # channel number gradually.
        adaptive_group_latency_change = self._cost_calculator.get_group_latency_contribute(
            self._group_mask, self._channel_group_size
        )

        grouped_neuron_idx_list = range(len(grouped_neurons))
        importance_list = [
            round(grouped_neurons[idx]["combined_importance"] * 1e6)
            for idx in grouped_neuron_idx_list
        ]

        layer_index_split = []
        cur_gn, cnt = grouped_neurons[0]["layer_group_name"], 0
        latency_list = [
            round(item * int_scale) for item in adaptive_group_latency_change[cur_gn]
        ]
        for idx in grouped_neuron_idx_list:
            if grouped_neurons[idx]["layer_group_name"] == cur_gn:
                cnt += 1
            else:
                layer_index_split.append(cnt)
                cur_gn, cnt = grouped_neurons[idx]["layer_group_name"], 1
                latency_list.extend(
                    [
                        round(item * int_scale)
                        for item in adaptive_group_latency_change[cur_gn]
                    ]
                )
        layer_index_split.append(cnt)

        items_in_bag, used_capacity, _ = _knapsack(
            weight=latency_list,
            value=importance_list,
            capacity=round(target_latency * int_scale),
            layer_index_split=layer_index_split,
            disable_layer_prune=self._disable_layer_prune,
        )
        used_capacity /= int_scale

        ori_channel_num = {}
        for group_name, mask in self._group_mask.items():
            ori_channel_num[group_name] = int(mask.sum().item())
        # Reset the mask to 0
        for group_name in self._group_mask.keys():
            self._group_mask[group_name][:] = 0.0
        # Set the selected neurons to active status
        for item_idx in items_in_bag:
            layer_group_name = grouped_neurons[item_idx]["layer_group_name"]
            channel_indices = grouped_neurons[item_idx]["channel_indices"]
            self._group_mask[layer_group_name][channel_indices] = 1.0

        total_pruned_num = 0
        for group_name, mask in self._group_mask.items():
            cur_channel_num = int(mask.sum().item())
            pruned_channel = ori_channel_num[group_name] - cur_channel_num
            total_pruned_num += pruned_channel * len(group_name)
            logger.info(
                "*** Group {}: {} channels / {} neurons are pruned at current step. "
                "{} channels / {} neurons left. ***".format(
                    group_name,
                    pruned_channel,
                    pruned_channel * len(group_name),
                    cur_channel_num,
                    cur_channel_num * len(group_name),
                )
            )
        # Mask conv and bn parameters
        self.mask_weights()
        # Get the total latency after pruning
        total_latency = self._cost_calculator.get_total_latency(self._group_mask)
        logger.debug(
            "Target latency: {}, Achieved latency: {}, "
            "Actual latency after pruning: {}".format(
                target_latency, used_capacity, total_latency
            )
        )
        return total_pruned_num


    def _group_neurons_by_rank(self):
        """Channel grouping for pruning.

        Group the neurons/channels in the same layer into groups to make the pruned
        structure GPU friendly.
        The channels are groupped according to the importance ranking.
        """
        grouped_neurons = []
        neuron_importance_scores = self._importance_calculator.neuron_metric
        for group, importance_value in neuron_importance_scores.items():
            if group in self._group_mask:
                mask = self._group_mask[group]
                value_remained = importance_value[mask == 1.0]
                channels_remained = np.arange(mask.size(0))[mask.cpu().numpy() == 1.0]
            else:
                self._group_mask[group] = torch.ones_like(importance_value)
                value_remained = importance_value
                channels_remained = np.arange(self._group_mask[group].size(0))

            sorted_values, sorted_indices = torch.sort(value_remained)
            pruning_group_size = max([self._channel_group_size[ln] for ln in group])
            sorted_values = sorted_values.view(-1, pruning_group_size)
            sorted_indices = sorted_indices.view(-1, pruning_group_size)
            combined_importance = sorted_values.sum(dim=1)
            # For resnet first layer, set to high importance score to avoid pruning.
            if "module.conv1" in group:
                combined_importance[:] = 10000
            for i in range(sorted_indices.size(0)):
                grouped_neurons.append(
                    {
                        "layer_group_name": group,
                        "channel_indices": [
                            channels_remained[idx] for idx in sorted_indices[i]
                        ],
                        "combined_importance": combined_importance[i].item(),
                    }
                )
        return grouped_neurons


    def mask_weights(self):
        """Zero the neuron weights according to the mask."""
        for group_name, mask in self._group_mask.items():
            for layer_name in group_name:
                layer = self.layers[layer_name]
                # Mask conv neurons
                layer.weight.data.mul_(mask.view(-1, 1, 1, 1))
                if layer.bias is not None:
                    layer.bias.data.mul_(mask)
                # Mask corresponding batch normalization layer
                layer_bn = self._conv_bn
                if (
                    layer_bn is not None
                    and layer_name in layer_bn
                    and layer_bn[layer_name] != ""
                ):
                    bn_layer = self.layers[layer_bn[layer_name]]
                    bn_layer.weight.data.mul_(mask)
                    bn_layer.bias.data.mul_(mask)
                    bn_layer.running_mean.data.mul_(mask)
                    bn_layer.running_var.data.mul_(mask)

        
    def print_neuron_num(self):
        logger.info("The number of channels remaining in each group:")
        for group, mask in self._group_mask.items():
            logger.info("{}: {}".format(group, int(mask.sum().cpu().item())))


def set_latency_prune_target(
    initial_latency: float,
    prune_start: int,
    prune_end: int,
    latency_reduction_ratio: float,
):
    """Schedule the pruning to achieve the target latency iteratively.

    Args:
        initial_latency (float): The latency of the full dense model.
        prune_start (int): the step to start pruning.
        prune_end (int): the step to stop pruning.
        latency_reduction_ratio (float): the latency reduction ratio realative to the
            initial total latency.
    Returns:
        latency_targets (list[float]): List with size of total (pruning) steps.
            Each item is the latency target (constraint) at the corresponding step.
    """
    target_latency = initial_latency * (1 - latency_reduction_ratio)
    # Exponential schedule proposed in https://arxiv.org/abs/2006.09081
    kt = [0 for _ in range(prune_end - prune_start + 1)]
    T = prune_end - prune_start
    for t in range(0, prune_end - prune_start + 1):
        alpha = t / T
        kt[t] = math.exp(
            alpha * math.log(target_latency) + (1 - alpha) * math.log(initial_latency)
        )
    to_prune = [kt[t] - kt[t + 1] for t in range(0, prune_end - prune_start)]
    latency_targets = []
    for item in to_prune:
        latency_targets.append(
            (latency_targets[-1] if len(latency_targets) > 0 else initial_latency)
            - item
        )
    return latency_targets


def _knapsack(weight, value, capacity, layer_index_split, disable_layer_prune=False):
    """Knapsack solver for neuron selection.

    Select the neurons maximizing the total importance score (value) within the
    given latency constraint (capacity).
    Args:
        weight (List[int]): The latency cost list of the grouped neurons.
        value (List[int]): The importance score list of the grouped neurons.
        capacity (Int): The given latency constraint.
        layer_index_split (List[int]): The smallest index of neurons belonging to each
            layer, used to indicate the start of a layer.
    Returns:
        items_in_bag (List[int]): The indices of the selected neuron groups to be kept.
        used_capacity (int): The total latency cost with the selected neuron groups.
        achieved_value (int): The achieved maximum total importance score.
    """
    if len(weight) != len(value):
        raise ValueError(
            "The given `weight` and `value` should have the same length. "
            f"Have length {len(weight)} and {len(value)} respectively."
        )
    if capacity >= sum(weight):
        return list(range(len(weight))), sum(weight), sum(value)

    neg_latency = [item for item in weight if item < 0]
    extra_space = 0 if len(neg_latency) == 0 else abs(sum(neg_latency))
    ori_capacity = capacity
    capacity += extra_space
    # To deal with the largest importance score in each layer first
    weight = weight[::-1]
    value = value[::-1]
    layer_index_split = [0] + list(itertools.accumulate(layer_index_split[::-1]))

    n_items = len(value)
    table = [[0.0] * (capacity + 1) for _ in range(2)]
    keep = [[False] * (capacity + 1) for _ in range(n_items + 1)]
    split_idx = 0
    for i in range(1, n_items + 1):
        wi = weight[i - 1]  # weight of current item
        vi = value[i - 1]  # value of current item
        index_old = (i - 1) % 2
        index_new = i % 2
        for w in range(capacity + 1):
            if w - wi > capacity:
                table[index_new][w] = table[index_old][w]
                continue
            val1 = vi + table[index_old][w - wi]
            val2 = table[index_old][w]
            # avoid layer pruning
            if disable_layer_prune:
                if layer_index_split[split_idx] == i - 1:
                    table[index_new][w] = val1
                    keep[i][w] = True
            get_larger_value = val1 > val2 or (val1 == val2 and wi == 0)
            meet_preceding_requirement = (
                keep[i - 1][w - wi] or layer_index_split[split_idx] == i - 1
            )
            if (
                # meet the requirement of capacity
                wi <= w
                # to get larger value
                and get_larger_value
                # to meet the preceding requirement
                and meet_preceding_requirement
            ):
                table[index_new][w] = val1
                keep[i][w] = True
            else:
                table[index_new][w] = val2
        if i - 1 == layer_index_split[split_idx]:
            split_idx += 1
    # retrieve the kept items
    items_in_bag = []
    K = ori_capacity
    for i in range(n_items, 0, -1):
        if keep[i][K]:
            items_in_bag.append(n_items - 1 - (i - 1))
            K -= weight[i - 1]
    used_capacity = ori_capacity - K
    achieved_value = table[n_items % 2][ori_capacity]

    return items_in_bag, used_capacity, achieved_value
