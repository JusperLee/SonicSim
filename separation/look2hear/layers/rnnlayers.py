###
# Author: Kai Li
# Date: 2021-07-12 17:31:32
# LastEditors: Please set LastEditors
# LastEditTime: 2021-09-17 09:26:46
###

import torch
import math
import inspect
from torch import nn
from torch import Tensor
from typing import Tuple
from typing import Optional
from torch.nn.functional import fold, unfold
import numpy as np

from . import activations, normalizations
from .normalizations import gLN


def has_arg(fn, name):
    """Checks if a callable accepts a given keyword argument.
    Args:
        fn (callable): Callable to inspect.
        name (str): Check if ``fn`` can be called with ``name`` as a keyword
            argument.
    Returns:
        bool: whether ``fn`` accepts a ``name`` keyword argument.
    """
    signature = inspect.signature(fn)
    parameter = signature.parameters.get(name)
    if parameter is None:
        return False
    return parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


class SingleRNN(nn.Module):
    """Module for a RNN block.
    Inspired from https://github.com/yluo42/TAC/blob/master/utility/models.py
    Licensed under CC BY-NC-SA 3.0 US.
    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        input_size (int): Dimension of the input feature. The input should have
            shape [batch, seq_len, input_size].
        hidden_size (int): Dimension of the hidden state.
        n_layers (int, optional): Number of layers used in RNN. Default is 1.
        dropout (float, optional): Dropout ratio. Default is 0.
        bidirectional (bool, optional): Whether the RNN layers are
            bidirectional. Default is ``False``.
    """

    def __init__(
        self,
        rnn_type,
        input_size,
        hidden_size,
        n_layers=1,
        dropout=0,
        bidirectional=False,
    ):
        super(SingleRNN, self).__init__()
        assert rnn_type.upper() in ["RNN", "LSTM", "GRU"]
        rnn_type = rnn_type.upper()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bool(bidirectional),
        )

    @property
    def output_size(self):
        return self.hidden_size * (2 if self.bidirectional else 1)

    def forward(self, inp):
        """ Input shape [batch, seq, feats] """
        self.rnn.flatten_parameters()  # Enables faster multi-GPU training.
        output = inp
        rnn_output, _ = self.rnn(output)
        return rnn_output


class LSTMBlockTF(nn.Module):
    def __init__(
        self,
        in_chan,
        hid_size,
        norm_type="gLN",
        bidirectional=True,
        rnn_type="LSTM",
        num_layers=1,
        dropout=0,
    ):
        super(LSTMBlockTF, self).__init__()
        self.RNN = SingleRNN(
            rnn_type,
            in_chan,
            hid_size,
            num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.linear = nn.Linear(self.RNN.output_size, in_chan)
        self.norm = normalizations.get(norm_type)(in_chan)

    def forward(self, x):
        B, F, T = x.size()
        output = self.RNN(x.transpose(1, 2))  # B, T, N
        output = self.linear(output)
        output = output.transpose(1, -1)  # B, N, T
        output = self.norm(output)
        return output + x


# ===================Transformer======================
class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class Swish(nn.Module):
    """
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs):
        return inputs * inputs.sigmoid()


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """

    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


class GLU(nn.Module):
    """
    The gating mechanism is called Gated Linear Units (GLU), which was first introduced for natural language processing
    in the paper “Language Modeling with Gated Convolutional Networks”
    """

    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class FeedForwardModule(nn.Module):
    def __init__(
        self, encoder_dim: int = 512, expansion_factor: int = 4, dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs):
        return self.sequential(inputs)


class PositionalEncoding(nn.Module):
    """
    Positional Encoding proposed in "Attention Is All You Need".
    Since transformer contains no recurrence and no convolution, in order for the model to make
    use of the order of the sequence, we must add some positional information.
    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """

    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout
    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """

    def __init__(
        self, d_model: int = 512, num_heads: int = 16, dropout_p: float = 0.1,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_embedding: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = (
            self.key_proj(key)
            .view(batch_size, -1, self.num_heads, self.d_head)
            .permute(0, 2, 1, 3)
        )
        value = (
            self.value_proj(value)
            .view(batch_size, -1, self.num_heads, self.d_head)
            .permute(0, 2, 1, 3)
        )
        pos_embedding = self.pos_proj(pos_embedding).view(
            batch_size, -1, self.num_heads, self.d_head
        )

        content_score = torch.matmul(
            (query + self.u_bias).transpose(1, 2), key.transpose(2, 3)
        )
        pos_score = torch.matmul(
            (query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1)
        )
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = torch.nn.functional.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(
            batch_size, num_heads, seq_length2 + 1, seq_length1
        )
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score


class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
    the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention
    module to generalize better on different input length and the resulting encoder is more robust to the variance of
    the utterance length. Conformer use prenorm residual units with dropout which helps training
    and regularizing deeper models.
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout
        device (torch.device): torch device (cuda or cpu)
    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """

    def __init__(
        self, d_model: int, num_heads: int, dropout_p: float = 0.1, is_casual=True
    ):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)
        self.is_casual = is_casual

    def forward(self, inputs: Tensor):
        batch_size, seq_length, _ = inputs.size()
        pos_embedding = self.positional_encoding(seq_length)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        mask = None
        if self.is_casual:
            mask = torch.triu(
                torch.ones((seq_length, seq_length), dtype=torch.uint8).to(
                    inputs.device
                ),
                diagonal=1,
            )
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1).bool()  # [B, L, L]

        inputs = self.layer_norm(inputs)
        outputs = self.attention(
            inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask
        )

        return self.dropout(outputs)


class ResidualConnectionModule(nn.Module):
    """
    Residual Connection Module.
    outputs = (module(inputs) x module_factor + inputs x input_factor)
    """

    def __init__(
        self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0
    ):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs):
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)


class DepthwiseConv1d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.
    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        is_casual: bool = True,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert (
            out_channels % in_channels == 0
        ), "out_channels should be constant multiple of in_channels"
        if is_casual:
            padding = kernel_size - 1
        else:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.is_casual = is_casual
        self.kernel_size = kernel_size

    def forward(self, inputs: Tensor) -> Tensor:
        if self.is_casual:
            return self.conv(inputs)[:, :, : -(self.kernel_size - 1)]
        return self.conv(inputs)


class PointwiseConv1d(nn.Module):
    """
    When kernel size == 1 conv1d, this operation is termed in literature as pointwise convolution.
    This operation often used to match dimensions.
    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by pointwise 1-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class ConformerConvModule(nn.Module):
    """
    Conformer convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is  deployed just after the convolution
    to aid training deep models.
    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout
        device (torch.device): torch device (cuda or cpu)
    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by conformer convolution module.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
        is_casual: bool = True,
    ) -> None:
        super(ConformerConvModule, self).__init__()
        assert (
            kernel_size - 1
        ) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape=(1, 2)),
            PointwiseConv1d(
                in_channels,
                in_channels * expansion_factor,
                stride=1,
                padding=0,
                bias=True,
            ),
            GLU(dim=1),
            DepthwiseConv1d(
                in_channels, in_channels, kernel_size, stride=1, is_casual=is_casual
            ),
            nn.BatchNorm1d(in_channels),
            Swish(),
            PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs).transpose(1, 2)


class TransformerLayer(nn.Module):
    def __init__(
        self, in_chan=128, n_head=8, n_att=1, dropout=0.1, max_len=500, is_casual=True
    ):
        super(TransformerLayer, self).__init__()
        self.in_chan = in_chan
        self.n_head = n_head
        self.dropout = dropout
        self.max_len = max_len
        self.n_att = n_att

        self.seq = nn.Sequential(
            ResidualConnectionModule(
                FeedForwardModule(in_chan, expansion_factor=4, dropout_p=dropout),
                module_factor=0.5,
            ),
            ResidualConnectionModule(
                MultiHeadedSelfAttentionModule(in_chan, n_head, dropout, is_casual)
            ),
            ResidualConnectionModule(
                ConformerConvModule(in_chan, 31, 2, dropout, is_casual=is_casual)
            ),
            ResidualConnectionModule(
                FeedForwardModule(in_chan, expansion_factor=4, dropout_p=dropout),
                module_factor=0.5,
            ),
            nn.LayerNorm(in_chan),
        )

    def forward(self, x):
        return self.seq(x)


class TransformerBlockTF(nn.Module):
    def __init__(
        self,
        in_chan,
        n_head=8,
        n_att=1,
        dropout=0.1,
        max_len=500,
        norm_type="cLN",
        is_casual=True,
    ):
        super(TransformerBlockTF, self).__init__()
        self.transformer = TransformerLayer(
            in_chan, n_head, n_att, dropout, max_len, is_casual
        )
        self.norm = normalizations.get(norm_type)(in_chan)

    def forward(self, x):
        B, F, T = x.size()
        output = self.transformer(x.permute(0, 2, 1).contiguous())  # B, T, N
        output = output.permute(0, 2, 1).contiguous()  # B, N, T
        output = self.norm(output)
        return output + x


# ====================================================


class DPRNNBlock(nn.Module):
    def __init__(
        self,
        in_chan,
        hid_size,
        norm_type="gLN",
        bidirectional=True,
        rnn_type="LSTM",
        num_layers=1,
        dropout=0,
    ):
        super(DPRNNBlock, self).__init__()
        self.intra_RNN = SingleRNN(
            rnn_type,
            in_chan,
            hid_size,
            num_layers,
            dropout=dropout,
            bidirectional=True,
        )
        self.inter_RNN = SingleRNN(
            rnn_type,
            in_chan,
            hid_size,
            num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.intra_linear = nn.Linear(self.intra_RNN.output_size, in_chan)
        self.intra_norm = normalizations.get(norm_type)(in_chan)

        self.inter_linear = nn.Linear(self.inter_RNN.output_size, in_chan)
        self.inter_norm = normalizations.get(norm_type)(in_chan)

    def forward(self, x):
        """ Input shape : [batch, feats, chunk_size, num_chunks] """
        B, N, K, L = x.size()
        output = x  # for skip connection
        # Intra-chunk processing
        x = x.transpose(1, -1).reshape(B * L, K, N)
        x = self.intra_RNN(x)
        x = self.intra_linear(x)
        x = x.reshape(B, L, K, N).transpose(1, -1)
        x = self.intra_norm(x)
        output = output + x
        # Inter-chunk processing
        x = output.transpose(1, 2).transpose(2, -1).reshape(B * K, L, N)
        x = self.inter_RNN(x)
        x = self.inter_linear(x)
        x = x.reshape(B, K, L, N).transpose(1, -1).transpose(2, -1).contiguous()
        x = self.inter_norm(x)
        return output + x


class DPRNN(nn.Module):
    def __init__(
        self,
        in_chan,
        n_src,
        out_chan=None,
        bn_chan=128,
        hid_size=128,
        chunk_size=100,
        hop_size=None,
        n_repeats=6,
        norm_type="gLN",
        mask_act="relu",
        bidirectional=True,
        rnn_type="LSTM",
        num_layers=1,
        dropout=0,
    ):
        super(DPRNN, self).__init__()
        self.in_chan = in_chan
        out_chan = out_chan if out_chan is not None else in_chan
        self.out_chan = out_chan
        self.bn_chan = bn_chan
        self.hid_size = hid_size
        self.chunk_size = chunk_size
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        self.n_src = n_src
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout

        layer_norm = normalizations.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)

        # Succession of DPRNNBlocks.
        net = []
        for x in range(self.n_repeats):
            net += [
                DPRNNBlock(
                    bn_chan,
                    hid_size,
                    norm_type=norm_type,
                    bidirectional=bidirectional,
                    rnn_type=rnn_type,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            ]
        self.net = nn.Sequential(*net)
        # Masking in 3D space
        net_out_conv = nn.Conv2d(bn_chan, n_src * bn_chan, 1)
        self.first_out = nn.Sequential(nn.PReLU(), net_out_conv)
        # Gating and masking in 2D space (after fold)
        self.net_out = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Sigmoid())
        self.mask_net = nn.Conv1d(bn_chan, out_chan, 1, bias=False)

        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, "dim"):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        r"""Forward.
        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$
        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, nsrc, nfilters, nframes)$
        """
        batch, n_filters, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)  # [batch, bn_chan, n_frames]
        output = unfold(
            output.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        n_chunks = output.shape[-1]
        output = output.reshape(batch, self.bn_chan, self.chunk_size, n_chunks)
        # Apply stacked DPRNN Blocks sequentially
        output = self.net(output)
        # Map to sources with kind of 2D masks
        output = self.first_out(output)
        output = output.reshape(
            batch * self.n_src, self.bn_chan, self.chunk_size, n_chunks
        )
        # Overlap and add:
        # [batch, out_chan, chunk_size, n_chunks] -> [batch, out_chan, n_frames]
        to_unfold = self.bn_chan * self.chunk_size
        output = fold(
            output.reshape(batch * self.n_src, to_unfold, n_chunks),
            (n_frames, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        # Apply gating
        output = output.reshape(batch * self.n_src, self.bn_chan, -1)
        # output = self.net_out(output) * self.net_gate(output)
        # Compute mask
        score = self.mask_net(output)
        est_mask = self.output_act(score)
        est_mask = est_mask.view(batch, self.n_src, self.out_chan, n_frames)
        return est_mask

    def get_config(self):
        config = {
            "in_chan": self.in_chan,
            "out_chan": self.out_chan,
            "bn_chan": self.bn_chan,
            "hid_size": self.hid_size,
            "chunk_size": self.chunk_size,
            "hop_size": self.hop_size,
            "n_repeats": self.n_repeats,
            "n_src": self.n_src,
            "norm_type": self.norm_type,
            "mask_act": self.mask_act,
            "bidirectional": self.bidirectional,
            "rnn_type": self.rnn_type,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }
        return config


class DPRNNLinear(nn.Module):
    def __init__(
        self,
        in_chan,
        n_src,
        out_chan=None,
        bn_chan=128,
        hid_size=128,
        chunk_size=100,
        hop_size=None,
        n_repeats=6,
        norm_type="gLN",
        mask_act="relu",
        bidirectional=True,
        rnn_type="LSTM",
        num_layers=1,
        dropout=0,
    ):
        super(DPRNNLinear, self).__init__()
        self.in_chan = in_chan
        out_chan = out_chan if out_chan is not None else in_chan
        self.out_chan = out_chan
        self.bn_chan = bn_chan
        self.hid_size = hid_size
        self.chunk_size = chunk_size
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        self.n_src = n_src
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout

        layer_norm = normalizations.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)

        # Succession of DPRNNBlocks.
        net = []
        for x in range(self.n_repeats):
            net += [
                DPRNNBlock(
                    bn_chan,
                    hid_size,
                    norm_type=norm_type,
                    bidirectional=bidirectional,
                    rnn_type=rnn_type,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            ]
        self.net = nn.Sequential(*net)
        # Masking in 3D space
        net_out_conv = nn.Conv2d(bn_chan, n_src * bn_chan, 1)
        self.first_out = nn.Sequential(nn.PReLU(), net_out_conv)
        # Gating and masking in 2D space (after fold)
        # self.net_out = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Tanh())
        self.net_out = nn.Linear(bn_chan, out_chan)
        self.net_gate = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Sigmoid())
        self.mask_net = nn.Conv1d(bn_chan, out_chan, 1, bias=False)

        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, "dim"):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        r"""Forward.
        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$
        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, nsrc, nfilters, nframes)$
        """
        batch, n_filters, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)  # [batch, bn_chan, n_frames]
        output = unfold(
            output.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        n_chunks = output.shape[-1]
        output = output.reshape(batch, self.bn_chan, self.chunk_size, n_chunks)
        # Apply stacked DPRNN Blocks sequentially
        output = self.net(output)
        # Map to sources with kind of 2D masks
        output = self.first_out(output)
        output = output.reshape(
            batch * self.n_src, self.bn_chan, self.chunk_size, n_chunks
        )
        # Overlap and add:
        # [batch, out_chan, chunk_size, n_chunks] -> [batch, out_chan, n_frames]
        to_unfold = self.bn_chan * self.chunk_size
        output = fold(
            output.reshape(batch * self.n_src, to_unfold, n_chunks),
            (n_frames, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        # Apply gating
        output = output.reshape(batch * self.n_src, self.bn_chan, -1)
        output = self.net_out(output.transpose(1, 1)).transpose(1, 2) * self.net_gate(
            output
        )
        # Compute mask
        score = self.mask_net(output)
        est_mask = self.output_act(score)
        est_mask = est_mask.view(batch, self.n_src, self.out_chan, n_frames)
        return est_mask

    def get_config(self):
        config = {
            "in_chan": self.in_chan,
            "out_chan": self.out_chan,
            "bn_chan": self.bn_chan,
            "hid_size": self.hid_size,
            "chunk_size": self.chunk_size,
            "hop_size": self.hop_size,
            "n_repeats": self.n_repeats,
            "n_src": self.n_src,
            "norm_type": self.norm_type,
            "mask_act": self.mask_act,
            "bidirectional": self.bidirectional,
            "rnn_type": self.rnn_type,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }
        return config
