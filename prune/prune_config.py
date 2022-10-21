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

import json


class PruneConfigReader(object):
    """Get model layer topology from config file for pruning."""
    def __init__(self, layer_prune_config):
        self._conv_bn = {}
        self._groups = []
        self._pre_group = {}
        self._layer_to_group = {}

        self._layer_prune_config = layer_prune_config
        self._set_prune_setting()

    def _set_prune_setting(self):
        """
            Set the regularization to the layers according to the configuration file
        Args:
            layer_prune_config:
        """
        with open(self._layer_prune_config, "r") as f:
            cf_dict = json.load(f)

        reg_groups = cf_dict["reg_groups"]
        for group in reg_groups:
            reg_type = group["reg_type"]
            assert reg_type in ["GS_SPARSE", "CL_GROUP"]
            for item in group["layers"]:
                name_list = item["layer_name"].replace(" ", "").split(",")
                bn_name_list = (item["bn_name"].replace(" ", "").split(",")
                                if item["bn_name"]
                                else ["" for _ in range(len(name_list))])
                pre_conv_name_list = (item["pre_conv"].replace(" ", "").split(",")
                                      if item["pre_conv"] else ["" for _ in range(len(name_list))])
                assert len(bn_name_list) == len(name_list)
                assert len(pre_conv_name_list) == len(name_list)
                for name, bn_name in zip(name_list, bn_name_list):
                    self._conv_bn[name] = bn_name

                if reg_type == "GS_SPARSE":
                    self._groups.extend([tuple([name]) for name in name_list])
                    for name, pre_conv_name in zip(name_list, pre_conv_name_list):
                        self._layer_to_group[name] = tuple([name])
                else:
                    self._groups.append(tuple(name_list))
                    for name in name_list:
                        self._layer_to_group[name] = self._groups[-1]

                for name, pre_conv_name in zip(name_list, pre_conv_name_list):
                    self._pre_group[name] = pre_conv_name

        for group_name, pre_conv_name in self._pre_group.items():
            self._pre_group[group_name] = self._layer_to_group[pre_conv_name] if pre_conv_name in self._layer_to_group else None

        has_bn = False
        for k, v in self._conv_bn.items():
            if v != "":
                has_bn = True
                break
        if not has_bn:
            self._conv_bn = None

    @property
    def prune_groups(self):
        """Get the list of layer groups where each group of layers are pruned together."""
        return self._groups
    @property
    def conv_bn(self):
        """Get the mapping from the Conv layer name to its following BN layer name."""
        return self._conv_bn
    @property
    def pre_group(self):
        """Get preceding layer group of a Conv layer.
        Get the mapping from the Conv layer name to the layer group that its preceding
        Conv layer belongs to.
        """
        return self._pre_group
