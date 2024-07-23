import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from .base_model import BaseModel

class G2Net(BaseModel):
    def __init__(self,
                 k1: list = [2, 3],
                 k2: list = [1, 3],
                 c: int = 64,
                 intra_connect: str = "cat",
                 d_feat: int = 256,
                 kd1: int = 3,
                 cd1: int = 64,
                 tcn_num: int = 2,
                 dilas: list = [1, 2, 5, 9],
                 fft_num: int = 320,
                 is_causal: bool = True,
                 acti_type: str = "sigmoid",
                 crm_type: str = "crm1",
                 stage_num: int = 3,
                 u_type: str = "u2",
                 head_type: str = "RI+MAG",
                 norm_type: str = "IN",  # switch to cLN leads to mild performance degradation but is still Ok. BN is the worst among the listed norm options.
                 n_fft: int = 512,
                 hop_length: int = 256,
                 win_length: int = 512,
                 sample_rate: int = 16000,
                 ):
        super(G2Net, self).__init__(sample_rate=sample_rate)
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        self.k1 = tuple(k1)
        self.k2 = tuple(k2)
        self.c = c
        self.intra_connect = intra_connect
        self.d_feat = d_feat
        self.kd1 = kd1
        self.cd1 = cd1
        self.tcn_num = tcn_num
        self.dilas = dilas
        self.fft_num = fft_num
        self.is_causal = is_causal
        self.acti_type = acti_type
        self.crm_type = crm_type
        self.stage_num = stage_num
        self.u_type = u_type
        self.head_type = head_type
        self.norm_type = norm_type
        stride = (1, 2)
        # components
        if u_type == "u2":
            if head_type == "RI":
                self.ri_en = U2Net_Encoder(2, self.k1, self.k2, stride, c, intra_connect, norm_type)
            elif head_type == "MAG":
                self.mag_en = U2Net_Encoder(1, self.k1, self.k2, stride, c, intra_connect, norm_type)
            elif head_type == "RI+MAG":
                self.ri_en = U2Net_Encoder(2, self.k1, self.k2, stride, c, intra_connect, norm_type)
                self.mag_en = U2Net_Encoder(1, self.k1, self.k2, stride, c, intra_connect, norm_type)
            elif head_type == "PHASE+MAG":
                self.phase_en = U2Net_Encoder(1, self.k1, self.k2, stride, c, intra_connect, norm_type)
                self.mag_en = U2Net_Encoder(1, self.k1, self.k2, stride, c, intra_connect, norm_type)
        elif u_type == "u":
            if head_type == "RI":
                self.ri_en = UNet_Encoder(2, self.k1, stride, c, norm_type)
            elif head_type == "MAG":
                self.mag_en = UNet_Encoder(1, self.k1, stride, c, norm_type)
            elif head_type == "RI+MAG":
                self.ri_en = UNet_Encoder(2, self.k1, stride, c, norm_type)
                self.mag_en = UNet_Encoder(1, self.k1, stride, c, norm_type)
            elif head_type == "PHASE+MAG":
                self.phase_en = UNet_Encoder(1, self.k1, stride, c, norm_type)
                self.mag_rn = UNet_Encoder(1, self.k1, stride, c, norm_type)

        ggm_block_list = []
        for i in range(stage_num):
            ggm_block_list.append(GGModule(d_feat,
                                           kd1,
                                           cd1,
                                           tcn_num,
                                           dilas,
                                           fft_num,
                                           is_causal,
                                           acti_type,
                                           crm_type,
                                           head_type,
                                           norm_type,
                                           ))
        self.ggms = nn.ModuleList(ggm_block_list)

    def forward(self, inpt) -> list:
        
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
        
        inpt_mag = torch.norm(batch_mix_stft, dim=1, keepdim=True)
        inpt_phase = torch.atan2(batch_mix_stft[:,-1,...], batch_mix_stft[:,0,...]).unsqueeze(dim=1)
        if self.head_type == "MAG":
            x = self.mag_en(inpt_mag)
            b_size, c, seq_len, _ = x.shape
            x = x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)
        elif self.head_type == "RI":
            x = self.ri_en(batch_mix_stft)
            b_size, c, seq_len, _ = x.shape
            x = x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)
        elif self.head_type == "RI+MAG":
            mag_x = self.mag_en(inpt_mag)
            ri_x = self.ri_en(batch_mix_stft)
            b_size, c, seq_len, _ = mag_x.shape
            mag_x = mag_x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)
            ri_x = ri_x.transpose(-2,-1).contiguous().view(b_size, -1, seq_len)
            x = torch.cat((ri_x, mag_x), dim=1)
        elif self.head_type == "PHASE+MAG":
            phase_x = self.phase_en(inpt_phase)
            mag_x = self.mag_en(inpt_mag)
            b_size, c, seq_len, _ = mag_x.shape
            phase_x = phase_x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)
            mag_x = mag_x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)
            x = torch.cat((phase_x, mag_x), dim=1)
        out_list = []
        pre_x = batch_mix_stft.transpose(-2, -1).contiguous()
        for i in range(self.stage_num):
            tmp = self.ggms[i](x, pre_x)
            pre_x = tmp
            out_list.append(pre_x)
        return out_list
    
    def get_model_args(self):
        model_args = {"n_src": 1}
        return model_args

class GGModule(nn.Module):
    def __init__(self,
                 d_feat: int,
                 kd1: int,
                 cd1: int,
                 tcn_num: int,
                 dilas: list,
                 fft_num: int,
                 is_causal: bool,
                 acti_type: str,
                 crm_type: str,
                 head_type: str,
                 norm_type: str,
                 ):
        super(GGModule, self).__init__()
        self.d_feat = d_feat
        self.kd1 = kd1
        self.cd1 = cd1
        self.tcn_num = tcn_num
        self.dilas = dilas
        self.fft_num = fft_num
        self.is_causal = is_causal
        self.acti_type = acti_type
        self.crm_type = crm_type
        self.head_type = head_type
        self.norm_type = norm_type

        # Components
        self.glance_branch = GlanceBranch(d_feat,kd1,cd1,tcn_num,dilas,fft_num,is_causal,acti_type,head_type,norm_type)
        self.gaze_branch = GazeBranch(d_feat,kd1,cd1,tcn_num,dilas,fft_num,is_causal,head_type,norm_type)

    def forward(self, x, pre_x):
        """
        :param x: (B,C1,T)
        :param pre_x: (B,2,C,T)
        :return: (B,2,C,T)
        """
        # pre_x: (B, 2, C, T)
        batch_num, _, c, seq_len = pre_x.size()
        pre_mag, pre_phase = torch.norm(pre_x, dim=1), torch.atan2(pre_x[:,-1,...], pre_x[:,0,...])
        pre_com = pre_x.view(batch_num, -1, seq_len)

        gain_filter = self.glance_branch(x, pre_mag)   # (B, C, T)
        com_resi = self.gaze_branch(x, pre_com)  # (B, 2, C, T)
        x_mag = pre_mag * gain_filter
        if self.crm_type == "crm1":  # crm1 yields better performance
            x_r, x_i = x_mag*torch.cos(pre_phase), x_mag*torch.sin(pre_phase)
            x = torch.stack((x_r, x_i), 1) + com_resi
        elif self.crm_type == "crm2":
            resi_phase = torch.atan2(com_resi[:,-1,...], com_resi[:,0,...])
            resi_mag = torch.norm(com_resi, dim=1)
            x_mag = x_mag + resi_mag
            x_phase = pre_phase + resi_phase
            x_r, x_i = x_mag * torch.cos(x_phase), x_mag * torch.sin(x_phase)
            x = torch.stack((x_r, x_i), 1)
        return x


class GlanceBranch(nn.Module):
    def __init__(self,
                 d_feat: int,
                 kd1: int,
                 cd1: int,
                 tcn_num: int,
                 dilas: list,
                 fft_num: int,
                 is_causal: bool,
                 acti_type: str,
                 head_type: str,
                 norm_type: str,
                 ):
        super(GlanceBranch, self).__init__()
        self.d_feat = d_feat
        self.kd1 = kd1
        self.cd1 = cd1
        self.tcn_num = tcn_num
        self.dilas = dilas
        self.fft_num = fft_num
        self.is_causal = is_causal
        self.acti_type = acti_type
        self.head_type = head_type
        self.norm_type = norm_type

        # Components
        if head_type == "RI" or head_type == "MAG":
            cin = (fft_num//2+1)+d_feat
        elif head_type == "RI+MAG" or head_type == "PHASE+MAG":
            cin = (fft_num//2+1)+d_feat*2
        else:
             raise Exception("Only RI, MAG, RI+MAG and PHASE+MAG are supported at present")
        self.in_conv = nn.Conv1d(cin, d_feat, 1)
        tcn_list = []
        for _ in range(tcn_num):
            tcn_list.append(SqueezedTCNList(d_feat, kd1, cd1, norm_type, dilas, is_causal))

        self.tcn_list = nn.ModuleList(tcn_list)
        self.linear_mag = nn.Conv1d(d_feat, fft_num//2+1, 1)
        if acti_type == "relu":
            self.acti = nn.ReLU()
        elif acti_type == "sigmoid":
            self.acti = nn.Sigmoid()
        elif acti_type == "tanh":
            self.acti = nn.Tanh()

    def forward(self, x, mag_x):
        """
        :param x: (B, C1, T)
        :param mag_x: (B, C2, T)
        :return: (B, C2, T)
        """
        x = torch.cat((x, mag_x), dim=1)
        x = self.in_conv(x)
        acc_x = torch.Tensor(torch.zeros(x.shape, requires_grad=True)).to(x.device)
        for i in range(len(self.tcn_list)):
            x = self.tcn_list[i](x)
            acc_x = acc_x + x
        x = self.acti(self.linear_mag(acc_x))
        return x

class GazeBranch(nn.Module):
    def __init__(self,
                 d_feat: int,
                 kd1: int,
                 cd1: int,
                 tcn_num: int,
                 dilas: list,
                 fft_num: int,
                 is_causal: bool,
                 head_type: str,
                 norm_type: str,
                 ):
        super(GazeBranch, self).__init__()
        self.d_feat = d_feat
        self.kd1 = kd1
        self.cd1 = cd1
        self.tcn_num = tcn_num
        self.dilas = dilas
        self.fft_num = fft_num
        self.is_causal = is_causal
        self.head_type = head_type
        self.norm_type = norm_type

        # Components
        if head_type == "RI" or head_type == "MAG":
            cin = (fft_num//2+1)*2+d_feat
        elif head_type == "RI+MAG" or head_type == "PHASE+MAG":
            cin = (fft_num//2+1)*2+d_feat*2
        else:
            raise Exception("Only RI, MAG, RI+MAG and PHASE+MAG are supported at present")
        self.in_conv_r = nn.Conv1d(cin, d_feat, 1)
        self.in_conv_i = nn.Conv1d(cin, d_feat, 1)
        tcn_list_r, tcn_list_i = [], []
        for _ in range(tcn_num):
            tcn_list_r.append(SqueezedTCNList(d_feat, kd1, cd1, norm_type, dilas, is_causal))
            tcn_list_i.append(SqueezedTCNList(d_feat, kd1, cd1, norm_type, dilas, is_causal))

        self.tcn_r = nn.ModuleList(tcn_list_r)
        self.tcn_i = nn.ModuleList(tcn_list_i)
        self.linear_r, self.linear_i = nn.Linear(d_feat, fft_num//2+1), nn.Linear(d_feat, fft_num//2+1)

    def forward(self, x, com_x):
        """
        x: the abstract feature from the branches, C1 = 256*2
        com_x: the flatten feature from the previous stage
        :param x: (B, C1, T)
        :param com_x: (B, C2, T)
        :return: (B,2,C,T)
        """
        x = torch.cat((x, com_x), dim=1)
        x_r, x_i = self.in_conv_r(x), self.in_conv_i(x)
        acc_r, acc_i = torch.Tensor(torch.zeros(x_r.shape, requires_grad=True)).to(x_r.device),\
                       torch.Tensor(torch.zeros(x_i.shape, requires_grad=True)).to(x_i.device)
        for i in range(len(self.tcn_r)):
            x_r, x_i = self.tcn_r[i](x_r), self.tcn_i[i](x_i)
            acc_r = acc_r + x_r
            acc_i = acc_i + x_i
        x = torch.stack((acc_r, acc_i), dim=1).transpose(-2,-1)  # (B,2,T,F)
        x_r, x_i = x[:,0,...], x[:,-1,...]
        x_r, x_i = self.linear_r(x_r).transpose(-2,-1), self.linear_i(x_i).transpose(-2,-1)
        return torch.stack((x_r, x_i), dim=1).contiguous()


class SqueezedTCNList(nn.Module):
    def __init__(self,
                 d_feat: int,
                 kd1: int,
                 cd1: int,
                 norm_type: str,
                 dilas: list = [1,2,5,9],
                 is_causal: bool = True):
        super(SqueezedTCNList, self).__init__()
        self.d_feat = d_feat
        self.kd1 = kd1
        self.cd1 = cd1
        self.norm_type = norm_type
        self.dilas = dilas
        self.is_causal = is_causal
        self.tcm_list = nn.ModuleList([SqueezedTCM(d_feat, kd1, cd1, dilas[i], is_causal, norm_type) for i in range(len(dilas))])

    def forward(self, x):
        for i in range(len(self.tcm_list)):
            x = self.tcm_list[i](x)
        return x

class SqueezedTCM(nn.Module):
    def __init__(self,
                 d_feat: int,
                 kd1: int,
                 cd1: int,
                 dilation: int,
                 is_causal: bool,
                 norm_type: str,
                 ):
        super(SqueezedTCM, self).__init__()
        self.d_feat = d_feat
        self.kd1 = kd1
        self.cd1 = cd1
        self.dilation = dilation
        self.is_causal = is_causal
        self.norm_type = norm_type
        if is_causal:
            pad = nn.ConstantPad1d(((kd1-1)*dilation, 0), value=0.)
        else:
            pad = nn.ConstantPad1d(((kd1-1)*dilation//2, (kd1-1)*dilation//2), value=0.)
        self.in_conv = nn.Conv1d(d_feat, cd1, kernel_size=1, bias=False)
        self.dd_conv_main = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            pad,
            nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False))
        self.dd_conv_gate = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            pad,
            nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False),
            nn.Sigmoid()
            )
        self.out_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            nn.Conv1d(cd1, d_feat, kernel_size=1, bias=False)
            )

    def forward(self, x):
        resi = x
        x = self.in_conv(x)
        x = self.dd_conv_main(x) * self.dd_conv_gate(x)
        x = self.out_conv(x)
        x = x + resi
        return x


class U2Net_Encoder(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: tuple,
                 k2: tuple,
                 stride: tuple,
                 c: int,
                 intra_connect: str,
                 norm_type: str,
                 ):
        super(U2Net_Encoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.k2 = k2
        self.stride = stride
        self.c = c
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        k_begin = (2, 5)
        c_end = 64

        meta_unet = []
        meta_unet.append(
            En_unet_module(cin, k_begin, k2, stride, c, intra_connect, norm_type, scale=4, de_flag=False, is_first=True))
        meta_unet.append(
            En_unet_module(cin, k1, k2, stride, c, intra_connect, norm_type, scale=3, de_flag=False))
        meta_unet.append(
            En_unet_module(cin, k1, k2, stride, c, intra_connect, norm_type, scale=2, de_flag=False))
        meta_unet.append(
            En_unet_module(cin, k1, k2, stride, c, intra_connect, norm_type, scale=1, de_flag=False))
        self.meta_unet_list = nn.ModuleList(meta_unet)
        self.last_conv = nn.Sequential(
            Gate2dconv(c, c_end, k1, stride, de_flag=False, pad=(0,0,k1[0]-1,0)),
            NormSwitch(norm_type, "2D", c_end),
            nn.PReLU(c_end)
        )

    def forward(self, x):
        for i in range(len(self.meta_unet_list)):
            x = self.meta_unet_list[i](x)
        x = self.last_conv(x)
        return x


class UNet_Encoder(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: tuple,
                 stride: tuple,
                 c: int,
                 norm_type: str,
                 ):
        super(UNet_Encoder, self).__init__()
        self.cin = cin
        self.k1, self.c = k1, c
        self.stride = stride
        self.norm_type = norm_type
        k_begin = (2, 5)
        c_end = 64  # 64 by default
        unet = []
        unet.append(nn.Sequential(
            Gate2dconv(cin, c, k_begin, stride, de_flag=False, pad=(0,0,k_begin[0]-1,0)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            Gate2dconv(c, c, k1, stride, de_flag=False, pad=(0,0,k1[0]-1,0)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            Gate2dconv(c, c, k1, stride, de_flag=False, pad=(0,0,k1[0]-1,0)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            Gate2dconv(c, c, k1, stride, de_flag=False, pad=(0,0,k1[0]-1,0)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            Gate2dconv(c, c_end, k1, stride, de_flag=False, pad=(0,0,k1[0]-1,0)),
            NormSwitch(norm_type, "2D", c_end),
            nn.PReLU(c_end)))
        self.unet_list = nn.ModuleList(unet)

    def forward(self, x):
        for i in range(len(self.unet_list)):
            x = self.unet_list[i](x)
        return x


class En_unet_module(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: tuple,
                 k2: tuple,
                 stride: tuple,
                 c: int,
                 intra_connect: str,
                 norm_type: str,
                 scale: int,
                 de_flag: bool = False,
                 is_first: bool = False,
                 ):
        super(En_unet_module, self).__init__()
        self.cin, self.k1, self.k2 = cin, k1, k2
        self.stride = stride
        self.c = c
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        self.scale = scale
        self.de_flag = de_flag
        self.is_first = is_first

        in_conv_list = []
        if self.is_first:
            in_conv_list.append(Gate2dconv(cin, c, k1, stride, de_flag, pad=(0, 0, k1[0]-1, 0)))
        else:
            in_conv_list.append(Gate2dconv(c, c, k1, stride, de_flag, pad=(0, 0, k1[0]-1, 0)))
        in_conv_list.append(NormSwitch(norm_type, "2D", c))
        in_conv_list.append(nn.PReLU(c))
        self.in_conv = nn.Sequential(*in_conv_list)

        enco_list, deco_list = [], []
        for _ in range(scale):
            enco_list.append(Conv2dunit(k2, stride, c, norm_type))
        for i in range(scale):
            if i == 0:
                deco_list.append(Deconv2dunit(k2, stride, c, "add", norm_type))
            else:
                deco_list.append(Deconv2dunit(k2, stride, c, intra_connect, norm_type))
        self.enco = nn.ModuleList(enco_list)
        self.deco = nn.ModuleList(deco_list)
        self.skip_connect = Skip_connect(intra_connect)

    def forward(self, x):
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
                 stride: tuple,
                 c: int,
                 norm_type: str,
                 ):
        super(Conv2dunit, self).__init__()
        self.k, self.c = k, c
        self.stride = stride
        self.norm_type = norm_type
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, k, stride),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)
        )

    def forward(self, x):
        return self.conv(x)


class Deconv2dunit(nn.Module):
    def __init__(self,
                 k: tuple,
                 stride: tuple,
                 c: int,
                 intra_connect: str,
                 norm_type: str,
                 ):
        super(Deconv2dunit, self).__init__()
        self.k, self.c = k, c
        self.stride = stride
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        deconv_list = []
        if self.intra_connect == "add":
            deconv_list.append(nn.ConvTranspose2d(c, c, k, stride))
        elif self.intra_connect == "cat":
            deconv_list.append(nn.ConvTranspose2d(2*c, c, k, stride))
        deconv_list.append(NormSwitch(norm_type, "2D", c))
        deconv_list.append(nn.PReLU(c))
        self.deconv = nn.Sequential(*deconv_list)

    def forward(self, x):
        return self.deconv(x)

class Gate2dconv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: tuple,
                 de_flag: bool,
                 pad: tuple = (0,0,0,0),
                 chomp=1,
                 ):
        super(Gate2dconv, self).__init__()
        if not de_flag:
            self.conv = nn.Sequential(
                nn.ConstantPad2d(pad, value=0.),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride))
            self.gate_conv = nn.Sequential(
                nn.ConstantPad2d(pad, value=0.),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
                nn.Sigmoid())
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
                Chomp_T(chomp))
            self.gate_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
                Chomp_T(chomp),
                nn.Sigmoid())

    def forward(self, x):
        return self.conv(x) * self.gate_conv(x)

class SelfAttention(nn.Module):
    def __init__(self, in_feat, d_feat, n_head=4, is_causal=True):
        super(SelfAttention, self).__init__()
        self.in_feat = in_feat
        self.d_feat = d_feat
        self.n_head = n_head
        self.is_causal = is_causal
        self.scale_factor = np.sqrt(d_feat//n_head)
        self.softmax = nn.Softmax(dim=-1)

        self.norm = nn.LayerNorm([in_feat])
        self.q_linear = nn.Linear(in_feat, d_feat)
        self.k_linear = nn.Linear(in_feat, d_feat)
        self.v_linear = nn.Linear(in_feat, d_feat)
        self.out_linear = nn.Linear(d_feat, in_feat)

    def Sequence_masl(self, seq):
        b_size, n_heads, seq_len, sub_d = seq.size()
        mask = torch.triu(torch.ones((b_size, n_heads, seq_len, seq_len), device=seq.device), diagonal=1)
        return mask

    def forward(self, x):
        """
        :param x: (B,Cin,T)
        :return: (B,F,T)
        """
        resi = x
        x = x.transpose(-2, -1).contiguous()
        x = self.norm(x)
        x_q = self.q_linear(x)
        x_k = self.k_linear(x)
        x_v = self.v_linear(x)

        b_size, seq_len, d_feat = x_q.shape
        x_q = x_q.view(b_size,seq_len,self.n_head,-1).transpose(1,2).contiguous()
        x_k = x_k.view(b_size,seq_len,self.n_head,-1).transpose(1,2).contiguous()
        x_v = x_v.view(b_size,seq_len,self.n_head,-1).transpose(1,2).contiguous()
        scores = torch.matmul(x_q, x_k.transpose(-2,-1)) / self.scale_factor
        if self.is_causal is True:
            scores = scores + (-1e9 * self.Sequence_masl(x_q))
        attn = self.softmax(scores)

        context = torch.matmul(attn, x_v)  # (B,N,T,D)
        context = context.permute(0,2,1,3).contiguous().view(b_size,seq_len,-1)
        context = self.out_linear(context).transpose(-2,-1).contiguous()
        return resi+context

class Conv1dunit(nn.Module):
    def __init__(self,
                 ci: int,
                 co: int,
                 k: int,
                 dila: int,
                 is_causal: bool,
                 norm_type: str,
                 ):
        super(Conv1dunit, self).__init__()
        self.ci, self.co, self.k, self.dila = ci, co, k, dila
        self.is_causal = is_causal
        if self.is_causal:
            pad = nn.ConstantPad1d(((k-1)*dila, 0), value=0.)
        else:
            pad = nn.ConstantPad1d(((k-1)*dila//2, (k-1)*dila//2), value=0.)

        self.unit = nn.Sequential(
            pad,
            nn.Conv1d(ci, co, k, dilation=dila),
            NormSwitch(norm_type, "1D", co),
            nn.PReLU(co)
        )
    def forward(self, x):
        x = self.unit(x)
        return x

class Skip_connect(nn.Module):
    def __init__(self, connect):
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
        return x[:, :, :-self.t, :]
    
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
