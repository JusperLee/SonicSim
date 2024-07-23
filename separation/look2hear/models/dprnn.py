import torch.nn.functional as F
from torch import nn
import torch
from .base_model import BaseModel


def select_norm(norm, dim, shape):
    return nn.GroupNorm(1, dim, torch.finfo(torch.float32).eps)


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
        # B x 1 x T -> B x C x T_out
        x = self.conv1d(x)
        x = F.relu(x)
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


class Dual_RNN_Block(nn.Module):
    """
    Implementation of the intra-RNN and the inter-RNN
    input:
         in_channels: The number of expected features in the input x
         out_channels: The number of features in the hidden state h
         rnn_type: RNN, LSTM, GRU
         norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
         dropout: If non-zero, introduces a Dropout layer on the outputs
                  of each LSTM layer except the last layer,
                  with dropout probability equal to dropout. Default: 0
         bidirectional: If True, becomes a bidirectional LSTM. Default: False
    """

    def __init__(
        self,
        out_channels,
        hidden_channels,
        rnn_type="LSTM",
        norm="ln",
        dropout=0,
        bidirectional=False,
        num_spks=2,
    ):
        super(Dual_RNN_Block, self).__init__()
        # RNN model
        self.intra_rnn = getattr(nn, rnn_type)(
            out_channels,
            hidden_channels,
            1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.inter_rnn = getattr(nn, rnn_type)(
            out_channels,
            hidden_channels,
            1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        # Norm
        self.intra_norm = select_norm(norm, out_channels, 4)
        self.inter_norm = select_norm(norm, out_channels, 4)
        # Linear
        self.intra_linear = nn.Linear(
            hidden_channels * 2 if bidirectional else hidden_channels, out_channels
        )
        self.inter_linear = nn.Linear(
            hidden_channels * 2 if bidirectional else hidden_channels, out_channels
        )

    def forward(self, x):
        """
        x: [B, N, K, S]
        out: [Spks, B, N, K, S]
        """
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        # [BS, K, H]
        intra_rnn, _ = self.intra_rnn(intra_rnn)
        # [BS, K, N]
        intra_rnn = self.intra_linear(intra_rnn.contiguous().view(B * S * K, -1)).view(
            B * S, K, -1
        )
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)

        # [B, N, K, S]
        intra_rnn = intra_rnn + x

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        # [BK, S, H]
        inter_rnn, _ = self.inter_rnn(inter_rnn)
        # [BK, S, N]
        inter_rnn = self.inter_linear(inter_rnn.contiguous().view(B * S * K, -1)).view(
            B * K, S, -1
        )
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out


class Dual_Path_RNN(nn.Module):
    """
    Implementation of the Dual-Path-RNN model
    input:
         in_channels: The number of expected features in the input x
         out_channels: The number of features in the hidden state h
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
        in_channels,
        out_channels,
        hidden_channels,
        rnn_type="LSTM",
        norm="ln",
        dropout=0,
        bidirectional=False,
        num_layers=4,
        K=200,
        num_spks=2,
    ):
        super(Dual_Path_RNN, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.dual_rnn = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_rnn.append(
                Dual_RNN_Block(
                    out_channels,
                    hidden_channels,
                    rnn_type=rnn_type,
                    norm=norm,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )
            )

        self.conv2d = nn.Conv2d(out_channels, out_channels * num_spks, kernel_size=1)
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1), nn.Tanh())
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [B, N, L]

        """
        # [B, N, L]
        x = self.norm(x)
        # [B, N, L]
        x = self.conv1d(x)
        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)
        # [B, N*spks, K, S]
        for i in range(self.num_layers):
            x = self.dual_rnn[i](x)
        x = self.prelu(x)
        x = self.conv2d(x)
        # [B*spks, N, K, S]
        B, _, K, S = x.shape
        x = x.view(B * self.num_spks, -1, K, S)
        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)
        # [spks*B, N, L]
        x = self.end_conv1x1(x)
        # [B*spks, N, L] -> [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)
        # [spks, B, N, L]
        x = x.transpose(0, 1)

        return x

    def _padding(self, input, K):
        """
        padding the audio times
        K: chunks of length
        P: hop size
        input: [B, N, L]
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        """
        the segmentation stage splits
        K: chunks of length
        P: hop size
        input: [B, N, L]
        output: [B, N, K, S]
        """
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(B, N, -1, K).transpose(2, 3)

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        """
        Merge sequence
        input: [B, N, K, S]
        gap: padding length
        output: [B, N, L]
        """
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


class DPRNNTasNet(BaseModel):
    """
    model of Dual Path RNN
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
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size=2,
        rnn_type="LSTM",
        norm="ln",
        dropout=0,
        bidirectional=False,
        num_layers=4,
        K=200,
        num_spks=2,
        sample_rate=16000,
    ):
        super(DPRNNTasNet, self).__init__(sample_rate=sample_rate)
        self.encoder = Encoder(kernel_size=kernel_size, out_channels=in_channels)
        self.separation = Dual_Path_RNN(
            in_channels,
            out_channels,
            hidden_channels,
            rnn_type=rnn_type,
            norm=norm,
            dropout=dropout,
            bidirectional=bidirectional,
            num_layers=num_layers,
            K=K,
            num_spks=num_spks,
        )
        self.decoder = Decoder(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            bias=False,
        )
        self.num_spks = num_spks

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
        s = self.separation(e)
        # [B, N, L] -> [B, L]
        out = [s[i] * e for i in range(self.num_spks)]
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
