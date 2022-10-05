''' Originated from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
'''

import pickle as pkl

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, layer_index, block_index, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, layer_num_dict=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        conv1_cout = layer_num_dict[f"layer{layer_index}.{block_index}.conv1"]
        conv2_cout = layer_num_dict[f"layer{layer_index}.{block_index}.conv2"]
        if conv1_cout == 0:
        # if channels[layer_index][block_index] == 0:
            self.conv1 = None
        else:
            self.conv1 = conv3x3(inplanes, conv1_cout, stride)
            self.bn1 = norm_layer(conv1_cout)
        self.relu = nn.ReLU(inplace=True)
        if self.conv1 is None:
            self.conv2 = None
            self.add_bias = nn.Parameter(torch.zeros(1, conv2_cout, 1, 1), requires_grad=False)
        else:
            self.conv2 = conv3x3(conv1_cout, conv2_cout)
            self.bn2 = norm_layer(conv2_cout)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        if self.conv1 is not None:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
        else:
            if self.downsample is not None:
                identity = self.downsample(x)
            out = self.add_bias + identity

        if self.conv1 is not None:
            out = self.relu(out)
        else:
            out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, layer_index, block_index, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, layer_num_dict=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        conv1_cout = layer_num_dict[f"layer{layer_index}.{block_index}.conv1"]
        conv2_cout = layer_num_dict[f"layer{layer_index}.{block_index}.conv2"]
        conv3_cout = layer_num_dict[f"layer{layer_index}.{block_index}.conv3"]
        if conv1_cout == 0 or conv2_cout == 0:
        # if channels[layer_index][block_index][0] == 0 or channels[layer_index][block_index][1] == 0:
            self.conv1 = None
        else:
            self.conv1 = conv1x1(inplanes, conv1_cout)
            self.bn1 = norm_layer(conv1_cout)
        if conv1_cout == 0 or conv2_cout == 0:
            self.conv2 = None
        else:
            self.conv2 = conv3x3(conv1_cout, conv2_cout, stride, groups, dilation)
            self.bn2 = norm_layer(conv2_cout)
        if self.conv1 is None or self.conv2 is None:
            self.conv3 = None
            self.add_bias = nn.Parameter(torch.zeros(1, conv3_cout, 1, 1), requires_grad=False)
        else:
            self.conv3 = conv1x1(conv2_cout, conv3_cout)
            self.bn3 = norm_layer(conv3_cout)
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        if self.conv1 is not None and self.conv2 is not None:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
        else:
            if self.downsample is not None:
                identity = self.downsample(x)
            out = self.add_bias + identity

        if self.conv1 is not None and self.conv2 is not None:
            out = self.relu(out)
        else:
            out = self.relu2(out)

        return out


class ResNet(nn.Module):

    def __init__(self, arch, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, layer_num_dict=None):
        super(ResNet, self).__init__()
        self.layers_have_downsample = []
        if arch == "resnet34":
            self.layers_have_downsample = [2, 3, 4]
        elif arch == "resnet50":
            self.layers_have_downsample = [1, 2, 3, 4]
        elif arch == "resnet101":
            self.layers_have_downsample = [1, 2, 3, 4]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = layer_num_dict["conv1"] if layer_num_dict is not None else 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 1, layers[0], layer_num_dict=layer_num_dict)
        self.layer2 = self._make_layer(block, 2, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       layer_num_dict=layer_num_dict)
        self.layer3 = self._make_layer(block, 3, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       layer_num_dict=layer_num_dict)
        self.layer4 = self._make_layer(block, 4, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       layer_num_dict=layer_num_dict)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        fc_cin = (
            layer_num_dict[f"layer4.{layers[-1]-1}.conv3"]
            if layer_num_dict is not None
            else 512 * block.expansion
        )
        self.fc = nn.Linear(fc_cin, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self, block, layer_index, blocks, stride=1, dilate=False, layer_num_dict=None
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if layer_index in self.layers_have_downsample:
            if layer_num_dict is not None:
                downsample_cout = layer_num_dict[f"layer{layer_index}.0.downsample.0"]
            else:
                planes = [64, 128, 256, 512]
                downsample_cout = planes[layer_index-1] * block.expansion
            downsample = nn.Sequential(
                conv1x1(self.inplanes, downsample_cout, stride),
                norm_layer(downsample_cout),
            )

        layers = []
        layers.append(block(self.inplanes, layer_index, 0, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, layer_num_dict))
        if block == BasicBlock:
            self.inplanes = layer_num_dict[f"layer{layer_index}.{blocks-1}.conv2"]
        else:
            self.inplanes = layer_num_dict[f"layer{layer_index}.{blocks-1}.conv3"]
        for i in range(1, blocks):
            layers.append(block(self.inplanes, layer_index, i, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, layer_num_dict=layer_num_dict))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def make_pruned_resnet(arch, group_mask_file, num_class=1000, enable_bias=False):
    with open(group_mask_file, "rb") as f:
        group_mask = pkl.load(f)
    
    layer_num_dict = {}
    for group, mask in group_mask.items():
        for layer_name in group:
            layer_num_dict[layer_name.replace("module.", "")] = int(mask.sum().item())

    arch = arch.lower()
    if arch in ["resnet18", "resnet34"]:
        building_block = BasicBlock
    elif arch in ["resnet50", "resnet101", "resnet152"]:
        building_block = Bottleneck
    else:
        raise ValueError(f"{arch} - unsupported resnet type")
    if arch == "resnet18":
        layer_block_num = [2, 2, 2, 2]
    elif arch == "resnet34":
        layer_block_num = [3, 4, 6, 3]
    elif arch == "resnet50":
        layer_block_num = [3, 4, 6, 3]
    elif arch == "resnet101":
        layer_block_num = [3, 4, 23, 3]
    elif arch == "resnet152":
        layer_block_num = [3, 8, 36, 3]

    model = ResNet(
        arch, building_block, layer_block_num, num_class, enable_bias, layer_num_dict=layer_num_dict
    )
    return model
