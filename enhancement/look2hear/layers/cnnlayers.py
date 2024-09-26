import torch
import torch.nn as nn
import torch.nn.functional as F
from . import normalizations, activations


class _Chop1d(nn.Module):
    """To ensure the output length is the same as the input."""

    def __init__(self, chop_size):
        super().__init__()
        self.chop_size = chop_size

    def forward(self, x):
        return x[..., : -self.chop_size].contiguous()


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        in_chan,
        hid_chan,
        skip_out_chan,
        kernel_size,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super(Conv1DBlock, self).__init__()
        self.skip_out_chan = skip_out_chan
        conv_norm = normalizations.get(norm_type)
        in_conv1d = nn.Conv1d(in_chan, hid_chan, 1)
        depth_conv1d = nn.Conv1d(
            hid_chan,
            hid_chan,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hid_chan,
        )
        if causal:
            depth_conv1d = nn.Sequential(depth_conv1d, _Chop1d(padding))
        self.shared_block = nn.Sequential(
            in_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan),
            depth_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan),
        )
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)
        if skip_out_chan:
            self.skip_conv = nn.Conv1d(hid_chan, skip_out_chan, 1)

    def forward(self, x):
        r"""Input shape $(batch, feats, seq)$."""
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        if not self.skip_out_chan:
            return res_out
        skip_out = self.skip_conv(shared_out)
        return res_out, skip_out


class ConvNormAct(nn.Module):
    """
    This class defines the convolution layer with normalization and a PReLU
    activation
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        kernel_size,
        stride=1,
        groups=1,
        dilation=1,
        padding=0,
        norm_type="gLN",
        act_type="prelu",
    ):
        super(ConvNormAct, self).__init__()
        self.conv = nn.Conv1d(
            in_chan,
            out_chan,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=True,
            groups=groups,
        )
        self.norm = normalizations.get(norm_type)(out_chan)
        self.act = activations.get(act_type)()

    def forward(self, x):
        output = self.conv(x)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        kernel_size,
        stride=1,
        groups=1,
        dilation=1,
        padding=0,
        norm_type="gLN",
    ):
        super(ConvNorm, self).__init__()
        self.conv = nn.Conv1d(
            in_chan,
            out_chan,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=True,
            groups=groups,
        )
        self.norm = normalizations.get(norm_type)(out_chan)

    def forward(self, x):
        output = self.conv(x)
        return self.norm(output)


class NormAct(nn.Module):
    """
    This class defines a normalization and PReLU activation
    """

    def __init__(
        self, out_chan, norm_type="gLN", act_type="prelu",
    ):
        """
        :param nOut: number of output channels
        """
        super(NormAct, self).__init__()
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = normalizations.get(norm_type)(out_chan)
        self.act = activations.get(act_type)()

    def forward(self, input):
        output = self.norm(input)
        return self.act(output)


class Video1DConv(nn.Module):
    """
    video part 1-D Conv Block
    in_chan: video Encoder output channels
    out_chan: dconv channels
    kernel_size: the depthwise conv kernel size
    dilation: the depthwise conv dilation
    residual: Whether to use residual connection
    skip_con: Whether to use skip connection
    first_block: first block, not residual
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        kernel_size,
        dilation=1,
        residual=True,
        skip_con=True,
        first_block=True,
    ):
        super(Video1DConv, self).__init__()
        self.first_block = first_block
        # first block, not residual
        self.residual = residual and not first_block
        self.bn = nn.BatchNorm1d(in_chan) if not first_block else None
        self.relu = nn.ReLU() if not first_block else None
        self.dconv = nn.Conv1d(
            in_chan,
            in_chan,
            kernel_size,
            groups=in_chan,
            dilation=dilation,
            padding=(dilation * (kernel_size - 1)) // 2,
            bias=True,
        )
        self.bconv = nn.Conv1d(in_chan, out_chan, 1)
        self.sconv = nn.Conv1d(in_chan, out_chan, 1)
        self.skip_con = skip_con

    def forward(self, x):
        """
        x: [B, N, T]
        out: [B, N, T]
        """
        if not self.first_block:
            y = self.bn(self.relu(x))
            y = self.dconv(y)
        else:
            y = self.dconv(x)
        # skip connection
        if self.skip_con:
            skip = self.sconv(y)
            if self.residual:
                y = y + x
                return skip, y
            else:
                return skip, y
        else:
            y = self.bconv(y)
            if self.residual:
                y = y + x
                return y
            else:
                return y


class Concat(nn.Module):
    def __init__(self, ain_chan, vin_chan, out_chan):
        super(Concat, self).__init__()
        self.ain_chan = ain_chan
        self.vin_chan = vin_chan
        # project
        self.conv1d = nn.Sequential(
            nn.Conv1d(ain_chan + vin_chan, out_chan, 1), nn.PReLU()
        )

    def forward(self, a, v):
        # up-sample video features
        v = torch.nn.functional.interpolate(v, size=a.size(-1))
        # concat: n x (A+V) x Ta
        y = torch.cat([a, v], dim=1)
        # conv1d
        return self.conv1d(y)


class FRCNNBlock(nn.Module):
    def __init__(
        self,
        in_chan=128,
        out_chan=512,
        upsampling_depth=4,
        norm_type="gLN",
        act_type="prelu",
    ):
        super().__init__()
        self.proj_1x1 = ConvNormAct(
            in_chan,
            out_chan,
            kernel_size=1,
            stride=1,
            groups=1,
            dilation=1,
            padding=0,
            norm_type=norm_type,
            act_type=act_type,
        )
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList([])
        self.spp_dw.append(
            ConvNorm(
                out_chan,
                out_chan,
                kernel_size=5,
                stride=1,
                groups=out_chan,
                dilation=1,
                padding=((5 - 1) // 2) * 1,
                norm_type=norm_type,
            )
        )
        # ----------Down Sample Layer----------
        for i in range(1, upsampling_depth):
            self.spp_dw.append(
                ConvNorm(
                    out_chan,
                    out_chan,
                    kernel_size=5,
                    stride=2,
                    groups=out_chan,
                    dilation=1,
                    padding=((5 - 1) // 2) * 1,
                    norm_type=norm_type,
                )
            )
        # ----------Fusion Layer----------
        self.fuse_layers = nn.ModuleList([])
        for i in range(upsampling_depth):
            fuse_layer = nn.ModuleList([])
            for j in range(upsampling_depth):
                if i == j:
                    fuse_layer.append(None)
                elif j - i == 1:
                    fuse_layer.append(None)
                elif i - j == 1:
                    fuse_layer.append(
                        ConvNorm(
                            out_chan,
                            out_chan,
                            kernel_size=5,
                            stride=2,
                            groups=out_chan,
                            dilation=1,
                            padding=((5 - 1) // 2) * 1,
                            norm_type=norm_type,
                        )
                    )
            self.fuse_layers.append(fuse_layer)
        self.concat_layer = nn.ModuleList([])
        # ----------Concat Layer----------
        for i in range(upsampling_depth):
            if i == 0 or i == upsampling_depth - 1:
                self.concat_layer.append(
                    ConvNormAct(
                        out_chan * 2,
                        out_chan,
                        1,
                        1,
                        norm_type=norm_type,
                        act_type=act_type,
                    )
                )
            else:
                self.concat_layer.append(
                    ConvNormAct(
                        out_chan * 3,
                        out_chan,
                        1,
                        1,
                        norm_type=norm_type,
                        act_type=act_type,
                    )
                )
        self.last_layer = nn.Sequential(
            ConvNormAct(
                out_chan * upsampling_depth,
                out_chan,
                1,
                1,
                norm_type=norm_type,
                act_type=act_type,
            )
        )
        self.res_conv = nn.Conv1d(out_chan, in_chan, 1)
        # ----------parameters-------------
        self.depth = upsampling_depth

    def forward(self, x):
        """
        :param x: input feature map
        :return: transformed feature map
        """
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            wav_length = output[i].shape[-1]
            y = torch.cat(
                (
                    self.fuse_layers[i][0](output[i - 1])
                    if i - 1 >= 0
                    else torch.Tensor().to(output1.device),
                    output[i],
                    F.interpolate(output[i + 1], size=wav_length, mode="nearest")
                    if i + 1 < self.depth
                    else torch.Tensor().to(output1.device),
                ),
                dim=1,
            )
            x_fuse.append(self.concat_layer[i](y))

        wav_length = output[0].shape[-1]
        for i in range(1, len(x_fuse)):
            x_fuse[i] = F.interpolate(x_fuse[i], size=wav_length, mode="nearest")

        concat = self.last_layer(torch.cat(x_fuse, dim=1))
        expanded = self.res_conv(concat)
        return expanded + residual


class Bottomup(nn.Module):
    def __init__(
        self,
        in_chan=128,
        out_chan=512,
        upsampling_depth=4,
        norm_type="gLN",
        act_type="prelu",
    ):
        super().__init__()
        self.proj_1x1 = ConvNormAct(
            in_chan,
            out_chan,
            kernel_size=1,
            stride=1,
            groups=1,
            dilation=1,
            padding=0,
            norm_type=norm_type,
            act_type=act_type,
        )
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList([])
        self.spp_dw.append(
            ConvNorm(
                out_chan,
                out_chan,
                kernel_size=5,
                stride=1,
                groups=out_chan,
                dilation=1,
                padding=((5 - 1) // 2) * 1,
                norm_type=norm_type,
            )
        )
        # ----------Down Sample Layer----------
        for i in range(1, upsampling_depth):
            self.spp_dw.append(
                ConvNorm(
                    out_chan,
                    out_chan,
                    kernel_size=5,
                    stride=2,
                    groups=out_chan,
                    dilation=1,
                    padding=((5 - 1) // 2) * 1,
                    norm_type=norm_type,
                )
            )

    def forward(self, x):
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        return residual, output[-1], output


class BottomupTCN(nn.Module):
    def __init__(
        self,
        in_chan=128,
        out_chan=512,
        upsampling_depth=4,
        norm_type="gLN",
        act_type="prelu",
    ):
        super().__init__()
        self.proj_1x1 = ConvNormAct(
            in_chan,
            out_chan,
            kernel_size=1,
            stride=1,
            groups=1,
            dilation=1,
            padding=0,
            norm_type=norm_type,
            act_type=act_type,
        )
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList([])
        self.spp_dw.append(
            Video1DConv(out_chan, out_chan, 3, skip_con=False, first_block=True)
        )
        # ----------Down Sample Layer----------
        for i in range(1, upsampling_depth):
            self.spp_dw.append(
                Video1DConv(out_chan, out_chan, 3, skip_con=False, first_block=False)
            )

    def forward(self, x):
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        return residual, output[-1], output


class Bottomup_Concat_Topdown(nn.Module):
    def __init__(
        self,
        in_chan=128,
        out_chan=512,
        upsampling_depth=4,
        norm_type="gLN",
        act_type="prelu",
    ):
        super().__init__()
        # ----------Fusion Layer----------
        self.fuse_layers = nn.ModuleList([])
        for i in range(upsampling_depth):
            fuse_layer = nn.ModuleList([])
            for j in range(upsampling_depth):
                if i == j:
                    fuse_layer.append(None)
                elif j - i == 1:
                    fuse_layer.append(None)
                elif i - j == 1:
                    fuse_layer.append(
                        ConvNorm(
                            out_chan,
                            out_chan,
                            kernel_size=5,
                            stride=2,
                            groups=out_chan,
                            dilation=1,
                            padding=((5 - 1) // 2) * 1,
                            norm_type=norm_type,
                        )
                    )
            self.fuse_layers.append(fuse_layer)
        self.concat_layer = nn.ModuleList([])
        # ----------Concat Layer----------
        for i in range(upsampling_depth):
            if i == 0 or i == upsampling_depth - 1:
                self.concat_layer.append(
                    ConvNormAct(
                        out_chan * 3,
                        out_chan,
                        1,
                        1,
                        norm_type=norm_type,
                        act_type=act_type,
                    )
                )
            else:
                self.concat_layer.append(
                    ConvNormAct(
                        out_chan * 4,
                        out_chan,
                        1,
                        1,
                        norm_type=norm_type,
                        act_type=act_type,
                    )
                )
        self.last_layer = nn.Sequential(
            ConvNormAct(
                out_chan * upsampling_depth,
                out_chan,
                1,
                1,
                norm_type=norm_type,
                act_type=act_type,
            )
        )
        self.res_conv = nn.Conv1d(out_chan, in_chan, 1)
        # ----------parameters-------------
        self.depth = upsampling_depth

    def forward(self, residual, bottomup, topdown):
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            wav_length = bottomup[i].shape[-1]
            y = torch.cat(
                (
                    self.fuse_layers[i][0](bottomup[i - 1])
                    if i - 1 >= 0
                    else torch.Tensor().to(bottomup[i].device),
                    bottomup[i],
                    F.interpolate(bottomup[i + 1], size=wav_length, mode="nearest")
                    if i + 1 < self.depth
                    else torch.Tensor().to(bottomup[i].device),
                    F.interpolate(topdown, size=wav_length, mode="nearest"),
                ),
                dim=1,
            )
            x_fuse.append(self.concat_layer[i](y))

        wav_length = bottomup[0].shape[-1]
        for i in range(1, len(x_fuse)):
            x_fuse[i] = F.interpolate(x_fuse[i], size=wav_length, mode="nearest")

        concat = self.last_layer(torch.cat(x_fuse, dim=1))
        expanded = self.res_conv(concat)
        return expanded + residual


class Bottomup_Concat_Topdown_TCN(nn.Module):
    def __init__(
        self,
        in_chan=128,
        out_chan=512,
        upsampling_depth=4,
        norm_type="gLN",
        act_type="prelu",
    ):
        super().__init__()
        # ----------Fusion Layer----------
        self.fuse_layers = nn.ModuleList([])
        for i in range(upsampling_depth):
            fuse_layer = nn.ModuleList([])
            for j in range(upsampling_depth):
                if i == j:
                    fuse_layer.append(None)
                elif j - i == 1:
                    fuse_layer.append(None)
                elif i - j == 1:
                    fuse_layer.append(None)
            self.fuse_layers.append(fuse_layer)
        self.concat_layer = nn.ModuleList([])
        # ----------Concat Layer----------
        for i in range(upsampling_depth):
            if i == 0 or i == upsampling_depth - 1:
                self.concat_layer.append(
                    ConvNormAct(
                        out_chan * 3,
                        out_chan,
                        1,
                        1,
                        norm_type=norm_type,
                        act_type=act_type,
                    )
                )
            else:
                self.concat_layer.append(
                    ConvNormAct(
                        out_chan * 4,
                        out_chan,
                        1,
                        1,
                        norm_type=norm_type,
                        act_type=act_type,
                    )
                )
        self.last_layer = nn.Sequential(
            ConvNormAct(
                out_chan * upsampling_depth,
                out_chan,
                1,
                1,
                norm_type=norm_type,
                act_type=act_type,
            )
        )
        self.res_conv = nn.Conv1d(out_chan, in_chan, 1)
        # ----------parameters-------------
        self.depth = upsampling_depth

    def forward(self, residual, bottomup, topdown):
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            wav_length = bottomup[i].shape[-1]
            y = torch.cat(
                (
                    bottomup[i - 1]
                    if i - 1 >= 0
                    else torch.Tensor().to(bottomup[i].device),
                    bottomup[i],
                    bottomup[i + 1]
                    if i + 1 < self.depth
                    else torch.Tensor().to(bottomup[i].device),
                    F.interpolate(topdown, size=wav_length, mode="nearest"),
                ),
                dim=1,
            )
            x_fuse.append(self.concat_layer[i](y))

        concat = self.last_layer(torch.cat(x_fuse, dim=1))
        expanded = self.res_conv(concat)
        return expanded + residual


class FRCNNBlockTCN(nn.Module):
    def __init__(
        self,
        in_chan=128,
        out_chan=512,
        upsampling_depth=4,
        norm_type="gLN",
        act_type="prelu",
    ):
        super().__init__()
        self.proj_1x1 = ConvNormAct(
            in_chan,
            out_chan,
            kernel_size=1,
            stride=1,
            groups=1,
            dilation=1,
            padding=0,
            norm_type=norm_type,
            act_type=act_type,
        )
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList([])
        self.spp_dw.append(
            Video1DConv(out_chan, out_chan, 3, skip_con=False, first_block=True)
        )
        # ----------Down Sample Layer----------
        for i in range(1, upsampling_depth):
            self.spp_dw.append(
                Video1DConv(out_chan, out_chan, 3, skip_con=False, first_block=False)
            )
        # ----------Fusion Layer----------
        self.fuse_layers = nn.ModuleList([])
        for i in range(upsampling_depth):
            fuse_layer = nn.ModuleList([])
            for j in range(upsampling_depth):
                if i == j:
                    fuse_layer.append(None)
                elif j - i == 1:
                    fuse_layer.append(None)
                elif i - j == 1:
                    fuse_layer.append(None)
            self.fuse_layers.append(fuse_layer)
        self.concat_layer = nn.ModuleList([])
        # ----------Concat Layer----------
        for i in range(upsampling_depth):
            if i == 0 or i == upsampling_depth - 1:
                self.concat_layer.append(
                    ConvNormAct(
                        out_chan * 2,
                        out_chan,
                        1,
                        1,
                        norm_type=norm_type,
                        act_type=act_type,
                    )
                )
            else:
                self.concat_layer.append(
                    ConvNormAct(
                        out_chan * 3,
                        out_chan,
                        1,
                        1,
                        norm_type=norm_type,
                        act_type=act_type,
                    )
                )
        self.last_layer = nn.Sequential(
            ConvNormAct(
                out_chan * upsampling_depth,
                out_chan,
                1,
                1,
                norm_type=norm_type,
                act_type=act_type,
            )
        )
        self.res_conv = nn.Conv1d(out_chan, in_chan, 1)
        # ----------parameters-------------
        self.depth = upsampling_depth

    def forward(self, x):
        """
        :param x: input feature map
        :return: transformed feature map
        """
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            wav_length = output[i].shape[-1]
            y = torch.cat(
                (
                    output[i - 1] if i - 1 >= 0 else torch.Tensor().to(output1.device),
                    output[i],
                    output[i + 1]
                    if i + 1 < self.depth
                    else torch.Tensor().to(output1.device),
                ),
                dim=1,
            )
            x_fuse.append(self.concat_layer[i](y))

        concat = self.last_layer(torch.cat(x_fuse, dim=1))
        expanded = self.res_conv(concat)
        return expanded + residual


class TAC(nn.Module):
    """Transform-Average-Concatenate inter-microphone-channel permutation invariant communication block [1].
    Args:
        input_dim (int): Number of features of input representation.
        hidden_dim (int, optional): size of hidden layers in TAC operations.
        activation (str, optional): type of activation used. See asteroid.masknn.activations.
        norm_type (str, optional): type of normalization layer used. See asteroid.masknn.norms.
    .. note:: Supports inputs of shape :math:`(batch, mic\_channels, features, chunk\_size, n\_chunks)`
        as in FasNet-TAC. The operations are applied for each element in ``chunk_size`` and ``n_chunks``.
        Output is of same shape as input.
    References
        [1] : Luo, Yi, et al. "End-to-end microphone permutation and number invariant multi-channel
        speech separation." ICASSP 2020.
    """

    def __init__(self, input_dim, hidden_dim=384, activation="prelu", norm_type="gLN"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_tf = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), activations.get(activation)()
        )
        self.avg_tf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), activations.get(activation)()
        )
        self.concat_tf = nn.Sequential(
            nn.Linear(2 * hidden_dim, input_dim), activations.get(activation)()
        )
        self.norm = normalizations.get(norm_type)(input_dim)

    def forward(self, x, valid_mics=None):
        """
        Args:
            x: (:class:`torch.Tensor`): Input multi-channel DPRNN features.
                Shape: :math:`(batch, mic\_channels, features, chunk\_size, n\_chunks)`.
            valid_mics: (:class:`torch.LongTensor`): tensor containing effective number of microphones on each batch.
                Batches can be composed of examples coming from arrays with a different
                number of microphones and thus the ``mic_channels`` dimension is padded.
                E.g. torch.tensor([4, 3]) means first example has 4 channels and the second 3.
                Shape:  :math`(batch)`.
        Returns:
            output (:class:`torch.Tensor`): features for each mic_channel after TAC inter-channel processing.
                Shape :math:`(batch, mic\_channels, features, chunk\_size, n\_chunks)`.
        """
        # Input is 5D because it is multi-channel DPRNN. DPRNN single channel is 4D.
        batch_size, nmics, channels, chunk_size, n_chunks = x.size()
        if valid_mics is None:
            valid_mics = torch.LongTensor([nmics] * batch_size)
        # First operation: transform the input for each frame and independently on each mic channel.
        output = self.input_tf(
            x.permute(0, 3, 4, 1, 2).reshape(
                batch_size * nmics * chunk_size * n_chunks, channels
            )
        ).reshape(batch_size, chunk_size, n_chunks, nmics, self.hidden_dim)

        # Mean pooling across channels
        if valid_mics.max() == 0:
            # Fixed geometry array
            mics_mean = output.mean(1)
        else:
            # Only consider valid channels in each batch element: each example can have different number of microphones.
            mics_mean = [
                output[b, :, :, : valid_mics[b]].mean(2).unsqueeze(0)
                for b in range(batch_size)
            ]  # 1, dim1*dim2, H
            mics_mean = torch.cat(mics_mean, 0)  # B*dim1*dim2, H

        # The average is processed by a non-linear transform
        mics_mean = self.avg_tf(
            mics_mean.reshape(batch_size * chunk_size * n_chunks, self.hidden_dim)
        )
        mics_mean = (
            mics_mean.reshape(batch_size, chunk_size, n_chunks, self.hidden_dim)
            .unsqueeze(3)
            .expand_as(output)
        )

        # Concatenate the transformed average in each channel with the original feats and
        # project back to same number of features
        output = torch.cat([output, mics_mean], -1)
        output = self.concat_tf(
            output.reshape(batch_size * chunk_size * n_chunks * nmics, -1)
        ).reshape(batch_size, chunk_size, n_chunks, nmics, -1)
        output = self.norm(
            output.permute(0, 3, 4, 1, 2).reshape(
                batch_size * nmics, -1, chunk_size, n_chunks
            )
        ).reshape(batch_size, nmics, -1, chunk_size, n_chunks)

        output += x
        return output
