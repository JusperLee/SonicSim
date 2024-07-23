import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_complex.tensor import ComplexTensor

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

from .base_model import BaseModel


EPS = torch.finfo(torch.get_default_dtype()).eps

class SingleLSTM(nn.Module):
    """Container module for a single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(
        self, input_size, hidden_size, dropout=0, bidirectional=False
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input, state=None):
        # input shape: batch, seq, dim
        # input = input.to(device)
        output = input
        rnn_output, state = self.rnn(output, state)
        rnn_output = self.dropout(rnn_output)
        rnn_output = self.proj(
            rnn_output.contiguous().view(-1, rnn_output.shape[2])
        ).view(output.shape)
        return rnn_output, state


def _pad_segment(input, segment_size):
    # input is the features: (B, N, T)
    batch_size, dim, seq_len = input.shape
    segment_stride = segment_size // 2

    rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
    if rest > 0:
        pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
        input = torch.cat([input, pad], 2)

    pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
    input = torch.cat([pad_aux, input, pad_aux], 2)

    return input, rest

def split_feature(input, segment_size):
    # split the feature into chunks of segment size
    # input is the features: (B, N, T)

    input, rest = _pad_segment(input, segment_size)
    batch_size, dim, seq_len = input.shape
    segment_stride = segment_size // 2

    segments1 = (
        input[:, :, :-segment_stride]
        .contiguous()
        .view(batch_size, dim, -1, segment_size)
    )
    segments2 = (
        input[:, :, segment_stride:]
        .contiguous()
        .view(batch_size, dim, -1, segment_size)
    )
    segments = (
        torch.cat([segments1, segments2], 3)
        .view(batch_size, dim, -1, segment_size)
        .transpose(2, 3)
    )

    return segments.contiguous(), rest


def merge_feature(input, rest):
    # merge the splitted features into full utterance
    # input is the features: (B, N, L, K)

    batch_size, dim, segment_size, _ = input.shape
    segment_stride = segment_size // 2
    input = (
        input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)
    )  # B, N, K, L

    input1 = (
        input[:, :, :, :segment_size]
        .contiguous()
        .view(batch_size, dim, -1)[:, :, segment_stride:]
    )
    input2 = (
        input[:, :, :, segment_size:]
        .contiguous()
        .view(batch_size, dim, -1)[:, :, :-segment_stride]
    )

    output = input1 + input2
    if rest > 0:
        output = output[:, :, :-rest]

    return output.contiguous()  # B, N, T


def choose_norm(norm_type, channel_size, shape="BDT"):
    """The input of normalization will be (M, C, K), where M is batch size.

    C is channel size and K is sequence length.
    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size, shape=shape)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size, shape=shape)
    elif norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)
    elif norm_type == "GN":
        return nn.GroupNorm(1, channel_size, eps=1e-8)
    else:
        raise ValueError("Unsupported normalization type")
    
class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)."""

    def __init__(self, channel_size, shape="BDT"):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()
        assert shape in ["BDT", "BTD"]
        self.shape = shape

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """Forward.

        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length

        Returns:
            gLN_y: [M, N, K]
        """
        if self.shape == "BTD":
            y = y.transpose(1, 2).contiguous()

        mean = y.mean(dim=(1, 2), keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=(1, 2), keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        if self.shape == "BTD":
            gLN_y = gLN_y.transpose(1, 2).contiguous()
        return gLN_y

class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)."""

    def __init__(self, channel_size, shape="BDT"):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()
        assert shape in ["BDT", "BTD"]
        self.shape = shape

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """Forward.

        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length

        Returns:
            cLN_y: [M, N, K]
        """

        assert y.dim() == 3

        if self.shape == "BTD":
            y = y.transpose(1, 2).contiguous()

        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        if self.shape == "BTD":
            cLN_y = cLN_y.transpose(1, 2).contiguous()

        return cLN_y


class Encoder(nn.Module):
    """
    Conv-Tasnet Encoder part
    kernel_size: the length of filters
    out_channels: the number of filters
    """

    def __init__(self, kernel_size=2, out_channels=64):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )

    def forward(self, x):
        """
        Input:
            x: [B, T], B is batch size, T is times
        Returns:
            x: [B, C, T_out]
            T_out is the number of time steps
        """
        # B x 1 x T -> B x C x T_out, then transpose C and T_out
        x = self.conv1d(x)
        x = F.relu(x)
        x = x.transpose(2, 1)
        return x


class Decoder(nn.ConvTranspose1d):
    """
    Decoder of the TasNet
    This module can be seen as the gradient of Conv1d with respect to its input.
    It is also known as a fractionally-strided convolution
    or a deconvolution (although it is not an actual deconvolution operation).
    """

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: [B, N, L]
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 3/4D tensor as input".format(self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        
        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        
        return x




class MemLSTM(nn.Module):
    """the Mem-LSTM of SkiM

    args:
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the LSTM layers are bidirectional.
            Default is False.
        mem_type: 'hc', 'h', 'c' or 'id'.
            It controls whether the hidden (or cell) state of
            SegLSTM will be processed by MemLSTM.
            In 'id' mode, both the hidden and cell states will
            be identically returned.
        norm_type: gLN, cLN. cLN is for causal implementation.
    """

    def __init__(
        self,
        hidden_size,
        dropout=0.0,
        bidirectional=False,
        mem_type="hc",
        norm_type="cLN",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.input_size = (int(bidirectional) + 1) * hidden_size
        self.mem_type = mem_type

        assert mem_type in [
            "hc",
            "h",
            "c",
            "id",
        ], f"only support 'hc', 'h', 'c' and 'id', current type: {mem_type}"

        if mem_type in ["hc", "h"]:
            self.h_net = SingleLSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            self.h_norm = choose_norm(
                norm_type=norm_type, channel_size=self.input_size, shape="BTD"
            )
        if mem_type in ["hc", "c"]:
            self.c_net = SingleLSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            self.c_norm = choose_norm(
                norm_type=norm_type, channel_size=self.input_size, shape="BTD"
            )

    def extra_repr(self) -> str:
        return f"Mem_type: {self.mem_type}, bidirectional: {self.bidirectional}"

    def forward(self, hc, S):
        # hc = (h, c), tuple of hidden and cell states from SegLSTM
        # shape of h and c: (d, B*S, H)
        # S: number of segments in SegLSTM

        if self.mem_type == "id":
            ret_val = hc
            h, c = hc
            d, BS, H = h.shape
            B = BS // S
        else:
            h, c = hc
            d, BS, H = h.shape
            B = BS // S
            h = h.transpose(1, 0).contiguous().view(B, S, d * H)  # B, S, dH
            c = c.transpose(1, 0).contiguous().view(B, S, d * H)  # B, S, dH
            if self.mem_type == "hc":
                h = h + self.h_norm(self.h_net(h)[0])
                c = c + self.c_norm(self.c_net(c)[0])
            elif self.mem_type == "h":
                h = h + self.h_norm(self.h_net(h)[0])
                c = torch.zeros_like(c)
            elif self.mem_type == "c":
                h = torch.zeros_like(h)
                c = c + self.c_norm(self.c_net(c)[0])

            h = h.view(B * S, d, H).transpose(1, 0).contiguous()
            c = c.view(B * S, d, H).transpose(1, 0).contiguous()
            ret_val = (h, c)

        if not self.bidirectional:
            # for causal setup
            causal_ret_val = []
            for x in ret_val:
                x = x.transpose(1, 0).contiguous().view(B, S, d * H)
                x_ = torch.zeros_like(x)
                x_[:, 1:, :] = x[:, :-1, :]
                x_ = x_.view(B * S, d, H).transpose(1, 0).contiguous()
                causal_ret_val.append(x_)
            ret_val = tuple(causal_ret_val)

        return ret_val

    def forward_one_step(self, hc, state):
        if self.mem_type == "id":
            pass
        else:
            h, c = hc
            d, B, H = h.shape
            h = h.transpose(1, 0).contiguous().view(B, 1, d * H)  # B, 1, dH
            c = c.transpose(1, 0).contiguous().view(B, 1, d * H)  # B, 1, dH
            if self.mem_type == "hc":
                h_tmp, state[0] = self.h_net(h, state[0])
                h = h + self.h_norm(h_tmp)
                c_tmp, state[1] = self.c_net(c, state[1])
                c = c + self.c_norm(c_tmp)
            elif self.mem_type == "h":
                h_tmp, state[0] = self.h_net(h, state[0])
                h = h + self.h_norm(h_tmp)
                c = torch.zeros_like(c)
            elif self.mem_type == "c":
                h = torch.zeros_like(h)
                c_tmp, state[1] = self.c_net(c, state[1])
                c = c + self.c_norm(c_tmp)
            h = h.transpose(1, 0).contiguous()
            c = c.transpose(1, 0).contiguous()
            hc = (h, c)

        return hc, state


class SegLSTM(nn.Module):

    """the Seg-LSTM of SkiM

    args:
        input_size: int, dimension of the input feature.
            The input should have shape (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the LSTM layers are bidirectional.
            Default is False.
        norm_type: gLN, cLN. cLN is for causal implementation.
    """

    def __init__(
        self, input_size, hidden_size, dropout=0.0, bidirectional=False, norm_type="cLN"
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)
        self.norm = choose_norm(
            norm_type=norm_type, channel_size=input_size, shape="BTD"
        )

    def forward(self, input, hc):
        # input shape: B, T, H

        B, T, H = input.shape

        if hc is None:
            # In fist input SkiM block, h and c are not available
            d = self.num_direction
            h = torch.zeros(d, B, self.hidden_size, dtype=input.dtype).to(input.device)
            c = torch.zeros(d, B, self.hidden_size, dtype=input.dtype).to(input.device)
        else:
            h, c = hc

        output, (h, c) = self.lstm(input, (h, c))
        output = self.dropout(output)
        output = self.proj(output.contiguous().view(-1, output.shape[2])).view(
            input.shape
        )
        output = input + self.norm(output)

        return output, (h, c)


class SkiM(nn.Module):
    """Skipping Memory Net

    args:
        input_size: int, dimension of the input feature.
            Input shape shoud be (batch, length, input_size)
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_blocks: number of basic SkiM blocks
        segment_size: segmentation size for splitting long features
        bidirectional: bool, whether the RNN layers are bidirectional.
        mem_type: 'hc', 'h', 'c', 'id' or None.
            It controls whether the hidden (or cell) state of SegLSTM
            will be processed by MemLSTM.
            In 'id' mode, both the hidden and cell states will
            be identically returned.
            When mem_type is None, the MemLSTM will be removed.
        norm_type: gLN, cLN. cLN is for causal implementation.
        seg_overlap: Bool, whether the segmentation will reserve 50%
            overlap for adjacent segments.Default is False.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout=0.0,
        num_blocks=2,
        segment_size=20,
        bidirectional=True,
        mem_type="hc",
        norm_type="gLN",
        seg_overlap=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.segment_size = segment_size
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.mem_type = mem_type
        self.norm_type = norm_type
        self.seg_overlap = seg_overlap

        assert mem_type in [
            "hc",
            "h",
            "c",
            "id",
            None,
        ], f"only support 'hc', 'h', 'c', 'id', and None, current type: {mem_type}"

        self.seg_lstms = nn.ModuleList([])
        for i in range(num_blocks):
            self.seg_lstms.append(
                SegLSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    norm_type=norm_type,
                )
            )
        if self.mem_type is not None:
            self.mem_lstms = nn.ModuleList([])
            for i in range(num_blocks - 1):
                self.mem_lstms.append(
                    MemLSTM(
                        hidden_size,
                        dropout=dropout,
                        bidirectional=bidirectional,
                        mem_type=mem_type,
                        norm_type=norm_type,
                    )
                )
        self.output_fc = nn.Sequential(
            nn.PReLU(), nn.Conv1d(input_size, output_size, 1)
        )

    def forward(self, input):
        # input shape: B, T (S*K), D
        B, T, D = input.shape

        if self.seg_overlap:
            input, rest = split_feature(
                input.transpose(1, 2), segment_size=self.segment_size
            )  # B, D, K, S
            input = input.permute(0, 3, 2, 1).contiguous()  # B, S, K, D
        else:
            input, rest = self._padfeature(input=input)
            input = input.view(B, -1, self.segment_size, D)  # B, S, K, D
        B, S, K, D = input.shape

        assert K == self.segment_size

        output = input.view(B * S, K, D).contiguous()  # BS, K, D
        hc = None
        for i in range(self.num_blocks):
            output, hc = self.seg_lstms[i](output, hc)  # BS, K, D
            if self.mem_type and i < self.num_blocks - 1:
                hc = self.mem_lstms[i](hc, S)
                pass

        if self.seg_overlap:
            output = output.view(B, S, K, D).permute(0, 3, 2, 1)  # B, D, K, S
            output = merge_feature(output, rest)  # B, D, T
            output = self.output_fc(output).transpose(1, 2)

        else:
            output = output.view(B, S * K, D)[:, :T, :]  # B, T, D
            output = self.output_fc(output.transpose(1, 2)).transpose(1, 2)

        return output

    def _padfeature(self, input):
        B, T, D = input.shape
        rest = self.segment_size - T % self.segment_size

        if rest > 0:
            input = torch.nn.functional.pad(input, (0, 0, 0, rest))
        return input, rest

    def forward_stream(self, input_frame, states):
        # input_frame # B, 1, N

        B, _, N = input_frame.shape

        def empty_seg_states():
            shp = (1, B, self.hidden_size)
            return (
                torch.zeros(*shp, device=input_frame.device, dtype=input_frame.dtype),
                torch.zeros(*shp, device=input_frame.device, dtype=input_frame.dtype),
            )

        B, _, N = input_frame.shape
        if not states:
            states = {
                "current_step": 0,
                "seg_state": [empty_seg_states() for i in range(self.num_blocks)],
                "mem_state": [[None, None] for i in range(self.num_blocks - 1)],
            }

        output = input_frame

        if states["current_step"] and (states["current_step"]) % self.segment_size == 0:
            tmp_states = [empty_seg_states() for i in range(self.num_blocks)]
            for i in range(self.num_blocks - 1):
                tmp_states[i + 1], states["mem_state"][i] = self.mem_lstms[
                    i
                ].forward_one_step(states["seg_state"][i], states["mem_state"][i])

            states["seg_state"] = tmp_states

        for i in range(self.num_blocks):
            output, states["seg_state"][i] = self.seg_lstms[i](
                output, states["seg_state"][i]
            )

        states["current_step"] += 1

        output = self.output_fc(output.transpose(1, 2)).transpose(1, 2)

        return output, states
    
    
class SkiMSeparator(torch.nn.Module):
    """Skipping Memory (SkiM) Separator

    Args:
        input_dim: input feature dimension
        causal: bool, whether the system is causal.
        num_spk: number of target speakers.
        nonlinear: the nonlinear function for mask estimation,
            select from 'relu', 'tanh', 'sigmoid'
        layer: int, number of SkiM blocks. Default is 3.
        unit: int, dimension of the hidden state.
        segment_size: segmentation size for splitting long features
        dropout: float, dropout ratio. Default is 0.
        mem_type: 'hc', 'h', 'c', 'id' or None.
            It controls whether the hidden (or cell) state of
            SegLSTM will be processed by MemLSTM.
            In 'id' mode, both the hidden and cell states
            will be identically returned.
            When mem_type is None, the MemLSTM will be removed.
        seg_overlap: Bool, whether the segmentation will reserve 50%
            overlap for adjacent segments. Default is False.
    """

    def __init__(
        self,
        input_dim: int,
        causal: bool = True,
        num_spk: int = 2,
        predict_noise: bool = False,
        nonlinear: str = "relu",
        layer: int = 3,
        unit: int = 512,
        segment_size: int = 20,
        dropout: float = 0.0,
        mem_type: str = "hc",
        seg_overlap: bool = False,
    ):
        super().__init__()

        self._num_spk = num_spk
        self.predict_noise = predict_noise

        self.segment_size = segment_size

        if mem_type not in ("hc", "h", "c", "id", None):
            raise ValueError("Not supporting mem_type={}".format(mem_type))

        self.num_outputs = self.num_spk + 1 if self.predict_noise else self.num_spk
        self.skim = SkiM(
            input_size=input_dim,
            hidden_size=unit,
            output_size=input_dim * self.num_outputs,
            dropout=dropout,
            num_blocks=layer,
            bidirectional=(not causal),
            norm_type="cLN" if causal else "gLN",
            segment_size=segment_size,
            seg_overlap=seg_overlap,
            mem_type=mem_type,
        )

        if nonlinear not in ("sigmoid", "relu", "tanh"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.nonlinear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
        }[nonlinear]

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        # ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, N]
            ilens (torch.Tensor): input lengths [Batch]
            additional (Dict or None): other data included in model
                NOTE: not used in this model

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, N), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """

        # if complex spectrum,
        if isinstance(input, ComplexTensor) or (not isinstance(input, ComplexTensor) and torch.is_complex(input)):
            feature = abs(input)
        else:
            feature = input

        B, T, N = feature.shape

        processed = self.skim(feature)  # B,T, N

        processed = processed.view(B, T, N, self.num_outputs)
        masks = self.nonlinear(processed).unbind(dim=3)
        if self.predict_noise:
            *masks, mask_noise = masks

        masked = [input * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        if self.predict_noise:
            others["noise1"] = input * mask_noise

        return masked, others  # , ilens in 2nd

    def forward_streaming(self, input_frame: torch.Tensor, states=None):
        if isinstance(input_frame, ComplexTensor) or (not isinstance(input_frame, ComplexTensor) and torch.is_complex(input_frame)):
            feature = abs(input_frame)
        else:
            feature = input_frame

        B, _, N = feature.shape

        processed, states = self.skim.forward_stream(feature, states=states)

        processed = processed.view(B, 1, N, self.num_outputs)
        masks = self.nonlinear(processed).unbind(dim=3)
        if self.predict_noise:
            *masks, mask_noise = masks

        masked = [input_frame * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        if self.predict_noise:
            others["noise1"] = input_frame * mask_noise

        return masked, states, others

    @property
    def num_spk(self):
        return self._num_spk
    

class SkiMNet(BaseModel):
    """
    model of SkiMNet
    input:
         in_channels: The number of expected features in the input x
         out_channels: The number of features in the hidden state h
         hidden_channels: The hidden size of RNN
         kernel_size: Encoder and Decoder Kernel size
         rnn_type: RNN, LSTM, GRU
         norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
         dropout: If non-zero, introduces a Dropout layer on the outputs
                  of each LSTM layer except the last layer,
                  with dropout probability equal to dropout. Default: 0
         bidirectional: If True, becomes a bidirectional LSTM. Default: False
         num_layers: number of Dual-Path-Block
         K: the length of chunk
         num_spks: the number of speakers
    """

    def __init__(
        self,
        input_dim,
        causal,
        num_spk,
        nonlinear,
        layer,
        unit,
        segment_size,
        dropout,
        mem_type,
        seg_overlap,
        kernel_size,
        sample_rate=8000
    ):
        super(SkiMNet, self).__init__(sample_rate=sample_rate)
        self.encoder = Encoder(kernel_size=kernel_size, out_channels=input_dim)
        self.separation = SkiMSeparator(
            input_dim=input_dim,
            causal=causal,
            num_spk=num_spk,
            predict_noise=False,
            nonlinear=nonlinear,
            layer=layer,
            unit=unit,
            segment_size=segment_size,
            dropout=dropout,
            mem_type=mem_type,
            seg_overlap=seg_overlap,
        )
        self.decoder = Decoder(
            in_channels=input_dim,
            out_channels=1,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            bias=False,
        )
        self.num_spks = num_spk
        self.causal = causal

    def forward(self, input):
        """
        x: [B, L]
        """
        # input shape: (B, C, T)
        was_one_d = False
        if input.ndim == 1:
            was_one_d = True
            input = input.unsqueeze(0).unsqueeze(1)
        if input.ndim == 2:
            was_one_d = True
            input = input.unsqueeze(1)
        if input.ndim == 3:
            input = input
        # [B, N, L]
        nsample = input.shape[-1]
        e = self.encoder(input)
        # [spks, B, N, L]
        if not self.causal:
            s, _ = self.separation(e)
        else:
            s, _ = self.separation(e)
            # s = []
            # state = None
            # import numpy as np
            # for i in range(e.shape[1]-1):
            #     frame = e[:, i : i + 1, :]
            #     frame_out, state, _ = self.separation.forward_streaming(frame, state)
            #     s.append((frame_out))
            #     print(frame_out)
            # exit(0)
        # [B, N, L] -> [B, L]
        out = [(s[i] * e).transpose(2, 1) for i in range(self.num_spks)]
        audio = [self.pad2(self.decoder(out[i]), nsample) for i in range(self.num_spks)]
        audio = torch.stack(audio, dim=1)
        return audio
    
    def pad2(self, input_tensor, target_len):
        input_tensor = torch.nn.functional.pad(
            input_tensor, (0, target_len - input_tensor.shape[-1])
        )
        return input_tensor

    def get_model_args(self):
        model_args = {"n_sample_rate": 2}
        return model_args
