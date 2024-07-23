import torch
import torch.nn as nn
from torch import Tensor
from .base_model import BaseModel

class GaGNet(BaseModel):
    def __init__(self,
                 cin: int,
                 k1: tuple,
                 k2: tuple,
                 c: int,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 p: int,
                 q: int,
                 dilas: list,
                 fft_num: int,
                 is_u2: bool,
                 is_causal: bool,
                 is_squeezed: bool,
                 acti_type: str,
                 intra_connect: str,
                 norm_type: str,
                 n_fft: int = 512,
                 hop_length: int = 256,
                 win_length: int = 512,
                 sample_rate: int = 16000,
                 ):
        """
        :param cin: input channels, default 2 for RI input
        :param k1: kernel size of 2-D GLU, (2, 3) by default
        :param k2: kernel size of the UNet-block, (1, 3) by default
        :param c: channels of the 2-D Convs, 64 by default
        :param kd1: kernel size of the dilated convs in the squeezedTCM, 3 by default
        :param cd1: channels of the dilated convs in the squeezedTCM, 64 by default
        :param d_feat: channels in the regular 1-D convs, 256 by default
        :param p: number of SqueezedTCMs within a group, 2 by default
        :param q: number of GGMs, 3 by default
        :param dilas: dilation rates, [1, 2, 5, 9] by default
        :param fft_num: fft number, 320 by default
        :param is_u2: whether U^{2} is set, True by default
        :param is_causal: whether causal setting, True by default
        :param is_squeezed: whether to squeeze the complex residual modeling path, False by default
        :param acti_type: the activation type in the glance block, "sigmoid" by default
        :param intra_connect: skip-connection type within the UNet-block , "cat" by default
        :param norm_type: "IN" by default.

        Be careful!!! track_running_stats in IN is False by default, i.e., in both training and validation phases, the
        statistics from the batch data are calculated, which causes the non-causality. For BN, however, track_running_sta
        ts is True by default, i.e., in the training phase, the batch statistics are calculated and in the inference
        phase, the global statistics will be fixed and lead to causal inference. If you choose IN, donot forget to switch
        the param track_running_stats into True!
        """
        super(GaGNet, self).__init__(sample_rate=sample_rate)
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        self.cin = cin
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.p = p
        self.q = q
        self.dilas = dilas
        self.fft_num = fft_num
        self.is_u2 = is_u2
        self.is_causal = is_causal
        self.is_squeezed = is_squeezed
        self.acti_type = acti_type
        self.intra_connect = intra_connect
        self.norm_type = norm_type

        if is_u2:
            self.en = U2Net_Encoder(cin, k1, k2, c, intra_connect, norm_type)
        else:
            self.en = UNet_Encoder(cin, k1, c, norm_type)
        self.gags = nn.ModuleList([GlanceGazeModule(kd1, cd1, d_feat, p, dilas, fft_num, is_causal, is_squeezed,
                                                    acti_type, norm_type) for _ in range(q)])

    def forward(self, inpt: Tensor) -> list:
        
        c = torch.sqrt(inpt.shape[-1] / torch.sum(inpt ** 2.0, dim=-1))
        inpt = inpt * c.unsqueeze(-1)
        
        batch_mix_stft = torch.stft(
            inpt,
            n_fft=self.fft_num,
            hop_length=self.hop_length,
            win_length=self.fft_num,
            window=torch.hann_window(self.fft_num).to(inpt.device),
            return_complex=True)  # (B,F,T,2)
        batch_mix_stft = torch.view_as_real(batch_mix_stft)
        # mix
        batch_mix_mag, batch_mix_phase = torch.norm(batch_mix_stft, dim=-1)**0.5, \
                                            torch.atan2(batch_mix_stft[..., -1], batch_mix_stft[..., 0])
        batch_mix_stft = torch.stack((batch_mix_mag*torch.cos(batch_mix_phase),
                                        batch_mix_mag*torch.sin(batch_mix_phase)), dim=-1)
        
        batch_mix_stft = batch_mix_stft.permute(0,3,2,1)  # (B,2,T,F)
        
        if inpt.ndim == 3:
            batch_mix_stft = batch_mix_stft.unsqueeze(dim=1)
        b_size, _, seq_len, _ = batch_mix_stft.shape
        feat_x = self.en(batch_mix_stft)
        x = feat_x.transpose(-2, -1).contiguous()
        x = x.view(b_size, -1, seq_len)
        pre_x = batch_mix_stft.transpose(-2, -1).contiguous()
        out_list = []
        for i in range(len(self.gags)):
            tmp = self.gags[i](x, pre_x)
            pre_x = tmp
            out_list.append(tmp)
        return out_list
    
    def get_model_args(self):
        model_args = {"n_src": 1}
        return model_args

class GlanceGazeModule(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 p: int,
                 dilas: list,
                 fft_num: int,
                 is_causal: bool,
                 is_squeezed: bool,
                 acti_type: str,
                 norm_type: str,
                 ):
        super(GlanceGazeModule, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.p = p
        self.dilas = dilas
        self.fft_num = fft_num
        self.is_causal = is_causal
        self.is_squeezed = is_squeezed
        self.acti_type = acti_type
        self.norm_type = norm_type

        self.glance_block = GlanceBlock(kd1, cd1, d_feat, p, dilas, fft_num, is_causal, acti_type, norm_type)
        self.gaze_block = GazeBlock(kd1, cd1, d_feat, p, dilas, fft_num, is_causal, is_squeezed, norm_type)

    def forward(self, feat_x: Tensor, pre_x: Tensor) -> Tensor:
        """
        :param feat_x: (B, C1, T)
        :param pre_x: (B, 2, F, T)
        :return: (B, 2, F, T)
        """
        gain_filter = self.glance_block(feat_x, pre_x)
        com_resi = self.gaze_block(feat_x, pre_x)
        # crm
        pre_mag, pre_phase = torch.norm(pre_x, dim=1), torch.atan2(pre_x[:,-1,...], pre_x[:,0,...])
        filtered_x = pre_mag * gain_filter
        coarse_x = torch.stack((filtered_x*torch.cos(pre_phase), filtered_x*torch.sin(pre_phase)), dim=1)  # coarse filtering
        x = coarse_x + com_resi
        return x


class GlanceBlock(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 p: int,
                 dilas: list,
                 fft_num: int,
                 is_causal: bool,
                 acti_type: str,
                 norm_type: str,
                 ):
        super(GlanceBlock, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.p = p
        self.dilas = dilas
        self.fft_num = fft_num
        self.is_causal = is_causal
        self.acti_type = acti_type
        self.norm_type = norm_type

        ci = (fft_num // 2 + 1) * 2 + d_feat
        self.in_conv_main = nn.Conv1d(ci, d_feat, 1)
        self.in_conv_gate = nn.Sequential(
            nn.Conv1d(ci, d_feat, 1),
            nn.Sigmoid()
        )
        tcn_g_list = []
        for _ in range(p):
            tcn_g_list.append(SqueezedTCNGroup(kd1, cd1, d_feat, dilas, is_causal, norm_type))
        self.tcn_g = nn.Sequential(*tcn_g_list)
        if acti_type == "sigmoid":
            acti = nn.Sigmoid()
        elif acti_type == "tanh":
            acti = nn.Tanh()
        elif acti_type == "relu":
            acti = nn.ReLU()
        else:
            raise RuntimeError("a activation function must be assigned!")
        self.linear_g = nn.Sequential(
            nn.Conv1d(d_feat, fft_num//2+1, 1),
            acti
        )

    def forward(self, feat_x: Tensor, pre_x: Tensor) -> Tensor:
        """
        :param feat_x: (B, C1, T)
        :param pre_x: (B, 2, F, T)
        :return: filter gain, (B, 1, F, T)
        """
        b_size, _, freq_num, seq_len = pre_x.shape
        pre_x = pre_x.view(b_size, -1, seq_len)
        inpt = torch.cat((feat_x, pre_x), dim=1)
        x = self.in_conv_main(inpt) * self.in_conv_gate(inpt)
        x = self.tcn_g(x)
        gain = self.linear_g(x)
        return gain


class GazeBlock(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 p: int,
                 dilas: list,
                 fft_num: int,
                 is_causal: bool,
                 is_squeezed: bool,
                 norm_type: str,
                 ):
        super(GazeBlock, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.p = p
        self.dilas = dilas
        self.fft_num = fft_num
        self.is_causal = is_causal
        self.is_squeezed = is_squeezed
        self.norm_type = norm_type

        # Components
        ci = (fft_num//2+1)*2 + d_feat
        self.in_conv_main = nn.Conv1d(ci, d_feat, 1)
        self.in_conv_gate = nn.Sequential(
            nn.Conv1d(ci, d_feat, 1),
            nn.Sigmoid()
        )

        if not is_squeezed:
            tcn_r_list, tcn_i_list = [], []
            for _ in range(p):
                tcn_r_list.append(SqueezedTCNGroup(kd1, cd1, d_feat, dilas, is_causal, norm_type))
                tcn_i_list.append(SqueezedTCNGroup(kd1, cd1, d_feat, dilas, is_causal, norm_type))
            self.tcm_r, self.tcm_i = nn.Sequential(*tcn_r_list), nn.Sequential(*tcn_i_list)
        else:
            tcn_ri_list = []
            for _ in range(p):
                tcn_ri_list.append(SqueezedTCNGroup(kd1, cd1, d_feat, dilas, is_causal, norm_type))
            self.tcm_ri = nn.Sequential(*tcn_ri_list)

        self.linear_r, self.linear_i = nn.Conv1d(d_feat, fft_num//2+1, 1), nn.Conv1d(d_feat, fft_num//2+1, 1)

    def forward(self, feat_x: Tensor, pre_x: Tensor) -> Tensor:
        """
        :param feat_x: (B, C1, T)
        :param pre_x:  (B, 2, F, T)
        :return: complex residual, (B, 2, F, T)
        """
        b_size, _, freq_num, seq_len = pre_x.shape
        pre_x = pre_x.view(b_size, -1, seq_len)
        inpt = torch.cat((feat_x, pre_x), dim=1)
        x = self.in_conv_main(inpt) * self.in_conv_gate(inpt)
        if not self.is_squeezed:
            x_r, x_i = self.tcm_r(x), self.tcm_i(x)
        else:
            x = self.tcm_ri(x)
            x_r, x_i = x, x
        x_r, x_i = self.linear_r(x_r), self.linear_i(x_i)
        return torch.stack((x_r, x_i), dim=1)


class SqueezedTCNGroup(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 dilas: list,
                 is_causal: bool,
                 norm_type: str,
                 ):
        super(SqueezedTCNGroup, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.dilas = dilas
        self.is_causal = is_causal
        self.nomr_type = norm_type

        tcn_list = []
        for i in range(len(dilas)):
            tcn_list.append(SqueezedTCM(kd1, cd1, d_feat, dilas[i], is_causal, norm_type))
        self.tcns = nn.Sequential(*tcn_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.tcns(x)


class SqueezedTCM(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 dilation: int,
                 is_causal: bool,
                 norm_type: str,
                 ):
        super(SqueezedTCM, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.dilation = dilation
        self.is_causal = is_causal
        self.norm_type = norm_type

        self.in_conv = nn.Conv1d(d_feat, cd1, 1, bias=False)
        if is_causal:
            padding = ((kd1-1)*dilation, 0)
        else:
            padding = ((kd1-1)*dilation//2, (kd1-1)*dilation//2)
        self.d_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            nn.ConstantPad1d(padding, value=0.),
            nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False)
            )
        self.out_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            nn.Conv1d(cd1, d_feat, 1, bias=False)
            )
    def forward(self, x: Tensor) -> Tensor:
        resi = x
        x = self.in_conv(x)
        x = self.d_conv(x)
        x = self.out_conv(x)
        x = x + resi
        return x


class U2Net_Encoder(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: tuple,
                 k2: tuple,
                 c: int,
                 intra_connect: str,
                 norm_type: str,
                 ):
        super(U2Net_Encoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        k_beg = (2, 5)
        c_end = 64
        meta_unet = []
        meta_unet.append(
            En_unet_module(cin, c, k_beg, k2, intra_connect, norm_type, scale=4))
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=3))
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=2))
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=1))
        self.meta_unet_list = nn.ModuleList(meta_unet)
        self.last_conv = nn.Sequential(
            GateConv2d(c, c_end, k1, (1,2)),
            NormSwitch(norm_type, "2D", c_end),
            nn.PReLU(c_end)
        )
    def forward(self, x: Tensor) -> Tensor:
        for i in range(len(self.meta_unet_list)):
            x = self.meta_unet_list[i](x)
        x = self.last_conv(x)
        return x


class UNet_Encoder(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: tuple,
                 c: int,
                 norm_type: str,
                 ):
        super(UNet_Encoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.c = c
        self.norm_type = norm_type
        k_beg = (2, 5)
        c_end = 64
        unet = []
        unet.append(nn.Sequential(
            GateConv2d(cin, c, k_beg, (1,2)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, (1,2)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, (1,2)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, (1,2)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c_end, k1, (1,2)),
            NormSwitch(norm_type, "2D", c_end),
            nn.PReLU(c_end)))
        self.unet_list = nn.ModuleList(unet)

    def forward(self, x: Tensor) -> Tensor:
        for i in range(len(self.unet_list)):
            x = self.unet_list[i](x)
        return x


class En_unet_module(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 k1: tuple,
                 k2: tuple,
                 intra_connect: str,
                 norm_type,
                 scale: int,
                 ):
        super(En_unet_module, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.cin = cin
        self.cout = cout
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        self.scale = scale

        in_conv_list = []
        in_conv_list.append(GateConv2d(cin, cout, k1, (1, 2)))
        in_conv_list.append(NormSwitch(norm_type, "2D", cout))
        in_conv_list.append(nn.PReLU(cout))
        self.in_conv = nn.Sequential(*in_conv_list)

        enco_list, deco_list = [], []
        for _ in range(scale):
            enco_list.append(Conv2dunit(k2, cout, norm_type))
        for i in range(scale):
            if i == 0:
                deco_list.append(Deconv2dunit(k2, cout, "add", norm_type))
            else:
                deco_list.append(Deconv2dunit(k2, cout, intra_connect, norm_type))
        self.enco = nn.ModuleList(enco_list)
        self.deco = nn.ModuleList(deco_list)
        self.skip_connect = Skip_connect(intra_connect)

    def forward(self, x: Tensor):
        x_resi = self.in_conv(x)
        x = x_resi
        x_list = []
        for i in range(len(self.enco)):
            x = self.enco[i](x)
            x_list.append(x)

        for i in range(len(self.deco)):
            if i == 0:
                x = self.deco[i](x)
            else:
                x_con = self.skip_connect(x, x_list[-(i+1)])
                x = self.deco[i](x_con)
        x_resi = x_resi + x
        del x_list
        return x_resi


class Conv2dunit(nn.Module):
    def __init__(self,
                 k: tuple,
                 c: int,
                 norm_type: str,
                 ):
        super(Conv2dunit, self).__init__()
        self.k = k
        self.c = c
        self.norm_type = norm_type
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, k, (1, 2)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c))

    def forward(self, x):
        return self.conv(x)


class Deconv2dunit(nn.Module):
    def __init__(self,
                 k: tuple,
                 c: int,
                 intra_connect: str,
                 norm_type: str,
                 ):
        super(Deconv2dunit, self).__init__()
        self.k = k
        self.c = c
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        deconv_list = []
        if self.intra_connect == "add":
            deconv_list.append(nn.ConvTranspose2d(c, c, k, (1, 2)))
        elif self.intra_connect == "cat":
            deconv_list.append(nn.ConvTranspose2d(2*c, c, k, (1, 2)))
        deconv_list.append(NormSwitch(norm_type, "2D", c))
        deconv_list.append(nn.PReLU(c))
        self.deconv = nn.Sequential(*deconv_list)

    def forward(self, x):
        return self.deconv(x)


class GateConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: tuple):
        super(GateConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        k_t = kernel_size[0]
        if k_t > 1:
            padding = (0, 0, k_t-1, 0)
            self.conv = nn.Sequential(
                nn.ConstantPad2d(padding, value=0.),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size, stride=stride))
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size,
                                  stride=stride)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(dim=1)
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * gate.sigmoid()

class Skip_connect(nn.Module):
    def __init__(self,
                 connect: str):
        super(Skip_connect, self).__init__()
        self.connect = connect

    def forward(self, x_main, x_aux):
        if self.connect == "add":
            x = x_main + x_aux
        elif self.connect == "cat":
            x = torch.cat((x_main, x_aux), dim=1)
        return x

class Chomp_T(nn.Module):
    def __init__(self,
                 t: int):
        super(Chomp_T, self).__init__()
        self.t = t

    def forward(self, x):
        return x[:, :, 0:-self.t, :]


class NormSwitch(nn.Module):
    """
    Currently, only BN and IN are considered
    """
    def __init__(self,
                 norm_type: str,
                 dim_size: str,
                 c: int,
                 ):
        super(NormSwitch, self).__init__()
        self.norm_type = norm_type
        self.dim_size = dim_size
        self.c = c

        assert norm_type in ["BN", "IN"] and dim_size in ["1D", "2D"]
        if norm_type == "BN":
            if dim_size == "1D":
                self.norm = nn.BatchNorm1d(c)
            else:
                self.norm = nn.BatchNorm2d(c)
        else:
            if dim_size == "1D":
                self.norm = nn.InstanceNorm1d(c, affine=True)
            else:
                self.norm = nn.InstanceNorm2d(c, affine=True)

    def forward(self, x):
        return self.norm(x)
