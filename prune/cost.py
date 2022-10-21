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

import pickle as pkl

import torch

class CostCalculator:
    """CostCalculator class getting the network and layer cost."""
    def __init__(
        self, layers, groups, pre_group, fmap_table, latency_lut_file, lut_bs
    ):
        self._layers = layers
        self._groups = groups
        self._pre_group = pre_group
        self._fmap_table = fmap_table
        
        with open(latency_lut_file, "rb") as f:
            self._latency_lut = pkl.load(f)
        self._lut_bs = lut_bs


    def get_total_latency(self, group_mask):
        """Get the approximated total latency for the conv network.

        Args:
            group_mask (Dict[Tuple[str,...], torch.Tensor]): Mapping the layer group name
                to the channel mask.
         Returns:
            total_latency (float): The total latency value.
        """
        total_latency = 0
        for group in self._groups:
            if group_mask is not None and group in group_mask:
                mask = group_mask[group]
                active_neuron_num = torch.sum(mask.data).cpu().item()
            else:
                active_neuron_num = self._layers[group[0]].weight.size(0)
            
            for layer_name in group:
                pre_group_name = self._pre_group[layer_name]
                if pre_group_name is None:
                    # For the first layer
                    pre_active_neuron_num  = 3
                elif group_mask is not None and pre_group_name in group_mask:
                    pre_active_neuron_num = torch.sum(
                        group_mask[pre_group_name].data).cpu().item()
                else:
                    pre_active_neuron_num = self._layers[pre_group_name[0]].weight.size(0)
                layer = self._layers[layer_name]
                k = layer.kernel_size[0]
                fmap = self._fmap_table[layer_name]
                stride = layer.stride[0]
                # Only general convolution and depth-wise convolution is supported
                group_count = pre_active_neuron_num if layer.groups > 1 else 1
                total_latency += self._get_layer_latency(
                    int(pre_active_neuron_num),
                    int(active_neuron_num),
                    int(k),
                    int(fmap),
                    int(stride),
                    int(group_count),
                )
        
        return total_latency


    def _get_layer_latency(self, cin, cout, kernel, feat_size, stride, groups):
        """Get the layer latency from the lookup table.

        Args:
            cin (int): Count of input channels
            cout (int): Count of output channels
            kernel (int): The kernel size
            feat_size (int): The input feature map size
            stride (int): The stride of the conv operation
            groups (int): The number of groups used in conv
        Returns:
            latency (float): The latency
        """
        if cin <= 0 or cout <= 0:
            return 0

        key = f"{self._lut_bs}_{cin}_{cout}_{feat_size}_{kernel}_{stride}_{groups}"
        if key not in self._latency_lut:
            raise ValueError(f"Configuration of {key} does not exist in latency lookup table.")
        latency = self._latency_lut[key]
        return latency


    def get_group_latency_contribute(self, group_mask, pruning_group_size):
        """Get the latency contribution of each neuron group.

        Args:
            group_mask (Dict[Tuple[str,...], torch.Tensor]): Mapping the layer group name
                to the channel mask.
            pruning_group_size (Dict[Tuple[str,...], int]): Indicate for each layer group the
                number of channels that needs to be grouped together.
        """
        group_latency_change = {}
        for group in self._groups:
            if group_mask is not None and group in group_mask:
                mask = group_mask[group]
                active_neuron_num = int(torch.sum(mask.data).cpu().item())
            else:
                active_neuron_num = int(self._layers[group[0]].weight.size(0))

            channel_group_size = max([pruning_group_size[ln] for ln in group])
            channel_group_count = active_neuron_num // channel_group_size
            latency_change = [0 for _ in range(channel_group_count)]
            # Latency change caused by the neuron num change in the pruned layers
            for layer_name in group:
                pre_group_name = self._pre_group[layer_name]
                if pre_group_name is None:
                    pre_active_neuron_num = 3
                elif group_mask is not None and pre_group_name in group_mask:
                    pre_active_neuron_num = int(
                        torch.sum(group_mask[pre_group_name].data).cpu().item()
                    )
                else:
                    pre_active_neuron_num = int(self._layers[pre_group_name[0]].weight.size(0))

                layer = self._layers[layer_name]
                k = layer.kernel_size[0]
                fmap = self._fmap_table[layer_name]
                stride = layer.stride[0]
                # Only general convolution and depth-wise convolution is supported
                conv_groups = pre_active_neuron_num if layer.groups > 1 else 1

                # Get adaptive latency contribution
                for i in range(channel_group_count):
                    latency = self._get_layer_latency(
                        cin=pre_active_neuron_num
                        if conv_groups == 1
                        else pre_active_neuron_num - channel_group_size * i,
                        cout=active_neuron_num - channel_group_size * i,
                        kernel=k,
                        feat_size=fmap,
                        stride=stride,
                        groups=conv_groups
                        if conv_groups == 1
                        else conv_groups - channel_group_size * i,
                    )
                    reduced_latency = self._get_layer_latency(
                        cin=pre_active_neuron_num
                        if conv_groups == 1
                        else max(pre_active_neuron_num - channel_group_size * (i + 1), 0),
                        cout=max(active_neuron_num - channel_group_size * (i + 1), 0),
                        kernel=k,
                        feat_size=fmap,
                        stride=stride,
                        groups=conv_groups
                        if conv_groups == 1
                        else max(conv_groups - channel_group_size * (i + 1), 0),
                    )
                    layer_latency_change = latency - reduced_latency
                    latency_change[i] += layer_latency_change

            group_latency_change[group] = latency_change

            # Latency change is calculated by decreasing the output channel num,
            # thus the first calculated latency change of this layer should
            # corresponds to the neuron(s) with least importance score.

        return group_latency_change




