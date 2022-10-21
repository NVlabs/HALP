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

import torch
import torch.nn as nn


class ChannelImportance:
    """For channel importance score calculation."""
    def __init__(self):
        self._neuron_metric = {}
        self._accumulate_count = 0


    def update_neuron_metric(self, layers, prune_groups, layer_bn=None):
        """Accumulate the neuron importance score for the layer groups to be pruned.

        Args:
            layers (Dict[str, torch.nn.Module]): Mapping the layer name to the corresponding
                layer module.
            prune_groups (List[Tuple[str, ...]]): A list of layer groups where each group of
                layers are pruned together. Each group of layer names is saved in a tuple.
            layer_bn (Dict[str, Any]): Mapping the name of a convolution layer to the name of
                its following batch normalization layer. Default is None, which means no
                batch normalization layer in the model.
        """
        cur_importance = {}
        for group in prune_groups:
            importance_score = self._calculate_first_order_taylor(layers, group, layer_bn)
            if (
                not torch.isnan(importance_score).any()
                and not torch.isinf(importance_score).any()
            ):
                cur_importance[group] = importance_score
            else:
                return
        for group in prune_groups:
            if group not in self._neuron_metric:
                self._neuron_metric[group] = cur_importance[group]
            else:
                self._neuron_metric[group] += cur_importance[group]
        self._accumulate_count += 1

    
    def get_average_importance(self):
        """Get the average importance score from the accumulation."""
        for group in self._neuron_metric.keys():
            # Average the score over past iterations.
            self._neuron_metric[group] /= self._accumulate_count


    def _calculate_first_order_taylor(self, layers, group, layer_bn):
        """First order of Taylor expansion importance calculation.

        Get the importance score for neuron using first order of Taylor expansion
        according to https://arxiv.org/abs/1906.10771.

        Args:
            layers (Dict[str, torch.nn.Module]): Mapping the layer name to the corresponding
                layer module.
            group (tuple[str]): The name of the layer group that requires an importance
                calculation.
            layer_bn (Dict[str, Any] or None): Mapping the name of a convolution layer to the
                name of its following batch normalization layer. Default is None, which means
                no batch normalization layer in the model. If batch normalization layer exists,
                the importance will be calculated on batch normalization weights; otherwise
                will be calculated on convolution weights.
        Returns:
            importance_score (torch.Tensor): The estimated importance of each channel for
                the given group.
        """
        # Check if there is any conv layer in the group having a batch normalization layer.
        if layer_bn is None:
            all_layer_has_bn = False
        else:
            all_layer_has_bn = True
            for layer_name in group:
                if (
                    layer_name not in layer_bn
                    or layer_bn[layer_name] is None
                    or layer_bn[layer_name] == ""
                ):
                    all_layer_has_bn = False
                    break
        if not all_layer_has_bn:
            # Calculate the importance on convolution weights
            layer = layers[group[0]]
            # Layers in the same group has the same channel number
            channel_num = layer.weight.size(0)
            weights = torch.empty((channel_num, 0), device=layer.weight.device)
            grads = torch.empty((channel_num, 0), device=layer.weight.device)
            for layer_name in group:
                layer = layers[layer_name]
                weights = torch.cat(
                    (weights, layer.weight.data.contiguous().view(channel_num, -1)),
                    dim=1,
                )
                grads = torch.cat(
                    (grads, layer.weight.grad.data.contiguous().view(channel_num, -1)),
                    dim=1,
                )
            importance_score = (weights * grads).sum(dim=1).abs()
        else:
            # Calculate the importance on batch normalization weights
            tmp_importance = 0
            for layer_name in group:
                layer = layers[layer_bn[layer_name]]
                tmp_importance += (
                    layer.weight.data * layer.weight.grad.data
                    + layer.bias.data * layer.bias.grad.data
                )
            importance_score = tmp_importance.abs()
        return importance_score

    
    def reset_importance(self):
        """Reset the neuron importance to 0."""
        for group in self._neuron_metric.keys():
            self._neuron_metric[group][:] = 0
        self._accumulate_count = 0


    @property
    def neuron_metric(self):
        """Get the neuron importance metric value."""
        return self._neuron_metric

