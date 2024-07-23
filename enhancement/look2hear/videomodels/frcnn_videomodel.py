###
# Author: Kai Li
# Date: 2021-06-29 01:19:16
# LastEditors: Please set LastEditors
# LastE
import sys

sys.path.append("../../")
import torch.nn as nn
import torch
from .shufflenetv2 import ShuffleNetV2
from .resnet1D import ResNet1D, BasicBlock1D
from .resnet import ResNet, BasicBlock
from torch.nn.modules.batchnorm import _BatchNorm


# -- auxiliary functions


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


def _average_batch(x, lengths, B):
    return torch.stack(
        [torch.mean(x[index][:, 0:i], 1) for index, i in enumerate(lengths)], 0
    )


class FRCNNVideoModel(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        backbone_type="resnet",
        relu_type="prelu",
        width_mult=1.0,
        pretrain=None,
    ):
        super(FRCNNVideoModel, self).__init__()
        self.backbone_type = backbone_type
        if self.backbone_type == "resnet":
            self.frontend_nout = 64
            self.backend_out = 512
            self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
        elif self.backbone_type == "shufflenet":
            assert width_mult in [0.5, 1.0, 1.5, 2.0], "Width multiplier not correct"
            shufflenet = ShuffleNetV2(input_size=96, width_mult=width_mult)
            self.trunk = nn.Sequential(
                shufflenet.features, shufflenet.conv_last, shufflenet.globalpool
            )
            self.frontend_nout = 24
            self.backend_out = 1024 if width_mult != 2.0 else 2048
            self.stage_out_channels = shufflenet.stage_out_channels[-1]

        frontend_relu = (
            nn.PReLU(num_parameters=self.frontend_nout)
            if relu_type == "prelu"
            else nn.ReLU()
        )
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                self.frontend_nout,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.pretrain = pretrain
        if pretrain:
            self.init_from(pretrain)

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = self.frontend3D(x)
        Tnew = x.shape[2]  # outpu should be B x C2 x Tnew x H x W
        x = threeD_to_2D_tensor(x)
        x = self.trunk(x)
        if self.backbone_type == "shufflenet":
            x = x.view(-1, self.stage_out_channels)
        x = x.view(B, Tnew, x.size(1))
        return x.transpose(1, 2)

    def init_from(self, path):
        pretrained_dict = torch.load(path, map_location="cpu")["model_state_dict"]
        update_frcnn_parameter(self, pretrained_dict)

    def train(self, mode=True):
        super().train(mode)
        if mode:  # freeze BN stats
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


def check_parameters(net):
    """
    Returns module parameters. Mb
    """
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10 ** 6


def update_frcnn_parameter(model, pretrained_dict):
    model_dict = model.state_dict()
    update_dict = {}
    for k, v in pretrained_dict.items():
        if "tcn" in k:
            pass
        else:
            update_dict[k] = v
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    for p in model.parameters():
        p.requires_grad = False
    return model


if __name__ == "__main__":
    frames = torch.randn(1, 1, 100, 96, 96)
