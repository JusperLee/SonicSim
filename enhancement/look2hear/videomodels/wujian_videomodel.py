###
# Author: Kai Li
# Date: 2021-06-21 12:03:35
# LastEditors: Kai Li
# LastEditTime: 2021-07-08 11:33:57
###

import torch.nn as nn
import math
import torch


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)


class BasicBlock(nn.Module):
    # Pytorch implementation
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # downsample 主要用来处理H(x)=F(x)+x中F(x)和xchannel维度不匹配问题
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # Pytorch implementation
    # remove the first conv
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super().__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.bnfc = nn.BatchNorm1d(num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.bnfc(x)
        return x


class front3D(nn.Module):
    """
    Video Encoder
    """

    def __init__(self, in_channels, out_channels, class_dim=256):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=class_dim)
        # self.resnet = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=class_dim)

    def forward(self, x):
        """
        x: [B, 1, T, H, W]
        out: [B, N, T]
        """
        # [B, N, T, H, W]
        x = self.conv3d(x)
        B, N, T, H, W = x.shape
        # [B, T, N, H, W]
        x = x.transpose(1, 2)
        # [BT, N, H, W]
        x = x.contiguous().view(B * T, N, H, W)
        # [BT, N]
        x = self.resnet(x)
        # [B, N, T]
        x = x.view(B, T, -1)
        x = x.transpose(1, 2).contiguous()
        return x


class WujianVideoModel(nn.Module):
    """
    All the Video Part
    in_channels: front3D part in_channels
    out_channels: front3D part out_channels and Video Conv1D part in_channels
    resnet_dim: the output dim of ResNet
    video_channels: conv1d output
    kernel_size: the kernel size of Video Conv1D
    repeat: Conv1D repeats
    """

    def __init__(self, prein_chan=1, preout_chan=64):
        super().__init__()
        self.front3d = front3D(prein_chan, preout_chan, class_dim=256)

    def forward(self, x):
        """
        x: [B, 1, T, H, W]
        out: [B, N, T]
        """
        # [B, N, T]
        x = self.front3d(x.unsqueeze(1))
        return x


def update_wujian_parameter(model, pretrained_dict):
    model_dict = model.state_dict()
    update_dict = {}
    for k, v in pretrained_dict.items():
        if "front3D" in k:
            k = k.split(".")
            k_ = "front3d.conv3d." + k[1] + "." + k[2]
            update_dict[k_] = v.clone()
        if "resnet" in k:
            k_ = "front3d." + k
            update_dict[k_] = v.clone()
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    for p in model.parameters():
        p.requires_grad = False
    return model
