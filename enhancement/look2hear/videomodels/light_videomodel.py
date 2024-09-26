###
# Author: Kai Li
# Date: 2022-03-16 03:46:10
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-03-16 04:07:39
###

import torch
import torch.nn as nn
import math
import numpy as np
from .shufflenetv2 import ShuffleNetV2


# -- auxiliary functions
def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


class LightVideomodel(nn.Module):
    def __init__(
        self, hidden_dim=256, relu_type="prelu", width_mult=1.0, pretrain=None
    ):
        super(LightVideomodel, self).__init__()

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

        # -- initialize
        self._initialize_weights_randomly()
        if pretrain:
            self.init_from(pretrain)

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = self.frontend3D(x)
        Tnew = x.shape[2]  # outpu should be B x C2 x Tnew x H x W
        x = threeD_to_2D_tensor(x)
        x = self.trunk(x)
        x = x.view(-1, self.stage_out_channels)
        x = x.view(B, Tnew, x.size(1))
        return x.transpose(1, 2)

    def _initialize_weights_randomly(self):

        use_sqrt = True

        if use_sqrt:

            def f(n):
                return math.sqrt(2.0 / float(n))

        else:

            def f(n):
                return 2.0 / float(n)

        for m in self.modules():
            if (
                isinstance(m, nn.Conv3d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
            ):
                n = np.prod(m.kernel_size) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif (
                isinstance(m, nn.BatchNorm3d)
                or isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.BatchNorm1d)
            ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))

    def init_from(self, path):
        pretrained_dict = torch.load(path, map_location="cpu")["model_state_dict"]
        update_light_parameter(self, pretrained_dict)


def update_light_parameter(model, pretrained_dict):
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


# if __name__ == "__main__":
#     frames = torch.randn(1, 1, 100, 88, 88)
#     model = LightLipreading(pretrain='/home/likai/nichang-stream/code/pretrain/lrw_snv1x_dsmstcn3x.pth.tar')
#     print(model(frames).shape)
