import warnings
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F


def make_enc_dec(
    fb_name,
    n_filters,
    kernel_size,
    stride=None,
    sample_rate=8000.0,
    who_is_pinv=None,
    padding=0,
    output_padding=0,
    **kwargs,
):
    """Creates congruent encoder and decoder from the same filterbank family.
    Args:
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``]. Can also be a class defined in a
            submodule in this subpackade (e.g. :class:`~.FreeFB`).
        n_filters (int): Number of filters.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        sample_rate (float): Sample rate of the expected audio.
            Defaults to 8000.0.
        who_is_pinv (str, optional): If `None`, no pseudo-inverse filters will
            be used. If string (among [``'encoder'``, ``'decoder'``]), decides
            which of ``Encoder`` or ``Decoder`` will be the pseudo inverse of
            the other one.
        padding (int): Zero-padding added to both sides of the input.
            Passed to Encoder and Decoder.
        output_padding (int): Additional size added to one side of the output shape.
            Passed to Decoder.
        **kwargs: Arguments which will be passed to the filterbank class
            additionally to the usual `n_filters`, `kernel_size` and `stride`.
            Depends on the filterbank family.
    Returns:
        :class:`.Encoder`, :class:`.Decoder`
    """
    fb_class = get(fb_name)

    if who_is_pinv in ["dec", "decoder"]:
        fb = fb_class(
            n_filters, kernel_size, stride=stride, sample_rate=sample_rate, **kwargs
        )
        enc = Encoder(fb, padding=padding)
        # Decoder filterbank is pseudo inverse of encoder filterbank.
        dec = Decoder.pinv_of(fb)
    elif who_is_pinv in ["enc", "encoder"]:
        fb = fb_class(
            n_filters, kernel_size, stride=stride, sample_rate=sample_rate, **kwargs
        )
        dec = Decoder(fb, padding=padding, output_padding=output_padding)
        # Encoder filterbank is pseudo inverse of decoder filterbank.
        enc = Encoder.pinv_of(fb)
    else:
        fb = fb_class(
            n_filters, kernel_size, stride=stride, sample_rate=sample_rate, **kwargs
        )
        enc = Encoder(fb, padding=padding)
        # Filters between encoder and decoder should not be shared.
        fb = fb_class(
            n_filters, kernel_size, stride=stride, sample_rate=sample_rate, **kwargs
        )
        dec = Decoder(fb, padding=padding, output_padding=output_padding)
    return enc, dec


def register_filterbank(custom_fb):
    """Register a custom filterbank, gettable with `filterbanks.get`.
    Args:
        custom_fb: Custom filterbank to register.
    """
    if (
        custom_fb.__name__ in globals().keys()
        or custom_fb.__name__.lower() in globals().keys()
    ):
        raise ValueError(
            f"Filterbank {custom_fb.__name__} already exists. Choose another name."
        )
    globals().update({custom_fb.__name__: custom_fb})


def get(identifier):
    """Returns a filterbank class from a string. Returns its input if it
    is callable (already a :class:`.Filterbank` for example).
    Args:
        identifier (str or Callable or None): the filterbank identifier.
    Returns:
        :class:`.Filterbank` or None
    """
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        cls = globals().get(identifier)
        if cls is None:
            raise ValueError(
                "Could not interpret filterbank identifier: " + str(identifier)
            )
        return cls
    else:
        raise ValueError(
            "Could not interpret filterbank identifier: " + str(identifier)
        )


class Filterbank(nn.Module):
    """Base Filterbank class.
    Each subclass has to implement a ``filters`` method.
    Args:
        n_filters (int): Number of filters.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the conv or transposed conv. (Hop size).
            If None (default), set to ``kernel_size // 2``.
        sample_rate (float): Sample rate of the expected audio.
            Defaults to 8000.
    Attributes:
        n_feats_out (int): Number of output filters.
    """

    def __init__(self, n_filters, kernel_size, stride=None, sample_rate=8000.0):
        super(Filterbank, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride if stride else self.kernel_size // 2
        # If not specified otherwise in the filterbank's init, output
        # number of features is equal to number of required filters.
        self.n_feats_out = n_filters
        self.sample_rate = sample_rate

    def filters(self):
        """Abstract method for filters."""
        raise NotImplementedError

    def pre_analysis(self, wav: torch.Tensor):
        """Apply transform before encoder convolution."""
        return wav

    def post_analysis(self, spec: torch.Tensor):
        """Apply transform to encoder convolution."""
        return spec

    def pre_synthesis(self, spec: torch.Tensor):
        """Apply transform before decoder transposed convolution."""
        return spec

    def post_synthesis(self, wav: torch.Tensor):
        """Apply transform after decoder transposed convolution."""
        return wav

    def get_config(self):
        """Returns dictionary of arguments to re-instantiate the class.
        Needs to be subclassed if the filterbanks takes additional arguments
        than ``n_filters`` ``kernel_size`` ``stride`` and ``sample_rate``.
        """
        config = {
            "fb_name": self.__class__.__name__,
            "n_filters": self.n_filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "sample_rate": self.sample_rate,
        }
        return config

    def forward(self, waveform):
        raise NotImplementedError(
            "Filterbanks must be wrapped with an Encoder or a Decoder."
        )


class _EncDec(nn.Module):
    """Base private class for Encoder and Decoder.
    Common parameters and methods.
    Args:
        filterbank (:class:`Filterbank`): Filterbank instance. The filterbank
            to use as an encoder or a decoder.
        is_pinv (bool): Whether to be the pseudo inverse of filterbank.
    Attributes:
        filterbank (:class:`Filterbank`)
        stride (int)
        is_pinv (bool)
    """

    def __init__(self, filterbank, is_pinv=False):
        super(_EncDec, self).__init__()
        self.filterbank = filterbank
        self.sample_rate = getattr(filterbank, "sample_rate", None)
        self.stride = self.filterbank.stride
        self.is_pinv = is_pinv

    def filters(self):
        return self.filterbank.filters()

    def compute_filter_pinv(self, filters):
        """Computes pseudo inverse filterbank of given filters."""
        scale = self.filterbank.stride / self.filterbank.kernel_size
        shape = filters.shape
        ifilt = torch.pinverse(filters.squeeze()).transpose(-1, -2).view(shape)
        # Compensate for the overlap-add.
        return ifilt * scale

    def get_filters(self):
        """Returns filters or pinv filters depending on `is_pinv` attribute"""
        if self.is_pinv:
            return self.compute_filter_pinv(self.filters())
        else:
            return self.filters()

    def get_config(self):
        """Returns dictionary of arguments to re-instantiate the class."""
        config = {"is_pinv": self.is_pinv}
        base_config = self.filterbank.get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Encoder(_EncDec):
    r"""Encoder class.
    Add encoding methods to Filterbank classes.
    Not intended to be subclassed.
    Args:
        filterbank (:class:`Filterbank`): The filterbank to use
            as an encoder.
        is_pinv (bool): Whether to be the pseudo inverse of filterbank.
        as_conv1d (bool): Whether to behave like nn.Conv1d.
            If True (default), forwarding input with shape :math:`(batch, 1, time)`
            will output a tensor of shape :math:`(batch, freq, conv\_time)`.
            If False, will output a tensor of shape :math:`(batch, 1, freq, conv\_time)`.
        padding (int): Zero-padding added to both sides of the input.
    """

    def __init__(self, filterbank, is_pinv=False, as_conv1d=True, padding=0):
        super(Encoder, self).__init__(filterbank, is_pinv=is_pinv)
        self.as_conv1d = as_conv1d
        self.n_feats_out = self.filterbank.n_feats_out
        self.kernel_size = self.filterbank.kernel_size
        self.padding = padding

    @classmethod
    def pinv_of(cls, filterbank, **kwargs):
        """Returns an :class:`~.Encoder`, pseudo inverse of a
        :class:`~.Filterbank` or :class:`~.Decoder`."""
        if isinstance(filterbank, Filterbank):
            return cls(filterbank, is_pinv=True, **kwargs)
        elif isinstance(filterbank, Decoder):
            return cls(filterbank.filterbank, is_pinv=True, **kwargs)

    def forward(self, waveform):
        """Convolve input waveform with the filters from a filterbank.
        Args:
            waveform (:class:`torch.Tensor`): any tensor with samples along the
                last dimension. The waveform representation with and
                batch/channel etc.. dimension.
        Returns:
            :class:`torch.Tensor`: The corresponding TF domain signal.
        Shapes
            >>> (time, ) -> (freq, conv_time)
            >>> (batch, time) -> (batch, freq, conv_time)  # Avoid
            >>> if as_conv1d:
            >>>     (batch, 1, time) -> (batch, freq, conv_time)
            >>>     (batch, chan, time) -> (batch, chan, freq, conv_time)
            >>> else:
            >>>     (batch, chan, time) -> (batch, chan, freq, conv_time)
            >>> (batch, any, dim, time) -> (batch, any, dim, freq, conv_time)
        """
        filters = self.get_filters()
        waveform = self.filterbank.pre_analysis(waveform)
        spec = multishape_conv1d(
            waveform,
            filters=filters,
            stride=self.stride,
            padding=self.padding,
            as_conv1d=self.as_conv1d,
        )
        return self.filterbank.post_analysis(spec)


def multishape_conv1d(
    waveform: torch.Tensor,
    filters: torch.Tensor,
    stride: int,
    padding: int = 0,
    as_conv1d: bool = True,
) -> torch.Tensor:
    if waveform.ndim == 1:
        # Assumes 1D input with shape (time,)
        # Output will be (freq, conv_time)
        return F.conv1d(
            waveform[None, None], filters, stride=stride, padding=padding
        ).squeeze()
    elif waveform.ndim == 2:
        # Assume 2D input with shape (batch or channels, time)
        # Output will be (batch or channels, freq, conv_time)
        warnings.warn(
            "Input tensor was 2D. Applying the corresponding "
            "Decoder to the current output will result in a 3D "
            "tensor. This behaviours was introduced to match "
            "Conv1D and ConvTranspose1D, please use 3D inputs "
            "to avoid it. For example, this can be done with "
            "input_tensor.unsqueeze(1)."
        )
        return F.conv1d(waveform.unsqueeze(1), filters, stride=stride, padding=padding)
    elif waveform.ndim == 3:
        batch, channels, time_len = waveform.shape
        if channels == 1 and as_conv1d:
            # That's the common single channel case (batch, 1, time)
            # Output will be (batch, freq, stft_time), behaves as Conv1D
            return F.conv1d(waveform, filters, stride=stride, padding=padding)
        else:
            # Return batched convolution, input is (batch, 3, time), output will be
            # (b, 3, f, conv_t). Useful for multichannel transforms. If as_conv1d is
            # false, (batch, 1, time) will output (batch, 1, freq, conv_time), useful for
            # consistency.
            return batch_packed_1d_conv(
                waveform, filters, stride=stride, padding=padding
            )
    else:  # waveform.ndim > 3
        # This is to compute "multi"multichannel convolution.
        # Input can be (*, time), output will be (*, freq, conv_time)
        return batch_packed_1d_conv(waveform, filters, stride=stride, padding=padding)


def batch_packed_1d_conv(
    inp: torch.Tensor, filters: torch.Tensor, stride: int = 1, padding: int = 0
):
    # Here we perform multichannel / multi-source convolution.
    # Output should be (batch, channels, freq, conv_time)
    batched_conv = F.conv1d(
        inp.view(-1, 1, inp.shape[-1]), filters, stride=stride, padding=padding
    )
    output_shape = inp.shape[:-1] + batched_conv.shape[-2:]
    return batched_conv.view(output_shape)


class Decoder(_EncDec):
    """Decoder class.
    Add decoding methods to Filterbank classes.
    Not intended to be subclassed.
    Args:
        filterbank (:class:`Filterbank`): The filterbank to use as an decoder.
        is_pinv (bool): Whether to be the pseudo inverse of filterbank.
        padding (int): Zero-padding added to both sides of the input.
        output_padding (int): Additional size added to one side of the
            output shape.
    .. note::
        ``padding`` and ``output_padding`` arguments are directly passed to
        ``F.conv_transpose1d``.
    """

    def __init__(self, filterbank, is_pinv=False, padding=0, output_padding=0):
        super().__init__(filterbank, is_pinv=is_pinv)
        self.padding = padding
        self.output_padding = output_padding

    @classmethod
    def pinv_of(cls, filterbank):
        """Returns an Decoder, pseudo inverse of a filterbank or Encoder."""
        if isinstance(filterbank, Filterbank):
            return cls(filterbank, is_pinv=True)
        elif isinstance(filterbank, Encoder):
            return cls(filterbank.filterbank, is_pinv=True)

    def forward(self, spec, length: Optional[int] = None) -> torch.Tensor:
        """Applies transposed convolution to a TF representation.
        This is equivalent to overlap-add.
        Args:
            spec (:class:`torch.Tensor`): 3D or 4D Tensor. The TF
                representation. (Output of :func:`Encoder.forward`).
            length: desired output length.
        Returns:
            :class:`torch.Tensor`: The corresponding time domain signal.
        """
        filters = self.get_filters()
        spec = self.filterbank.pre_synthesis(spec)
        wav = multishape_conv_transpose1d(
            spec,
            filters,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        wav = self.filterbank.post_synthesis(wav)
        if length is not None:
            length = min(length, wav.shape[-1])
            return wav[..., :length]
        return wav


def multishape_conv_transpose1d(
    spec: torch.Tensor,
    filters: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
) -> torch.Tensor:
    if spec.ndim == 2:
        # Input is (freq, conv_time), output is (time)
        return F.conv_transpose1d(
            spec.unsqueeze(0),
            filters,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        ).squeeze()
    if spec.ndim == 3:
        # Input is (batch, freq, conv_time), output is (batch, 1, time)
        return F.conv_transpose1d(
            spec,
            filters,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
    else:
        # Multiply all the left dimensions together and group them in the
        # batch. Make the convolution and restore.
        view_as = (-1,) + spec.shape[-2:]
        out = F.conv_transpose1d(
            spec.reshape(view_as),
            filters,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        return out.view(spec.shape[:-2] + (-1,))


class FreeFB(Filterbank):
    """Free filterbank without any constraints. Equivalent to
    :class:`nn.Conv1d`.
    Args:
        n_filters (int): Number of filters.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        sample_rate (float): Sample rate of the expected audio.
            Defaults to 8000.
    Attributes:
        n_feats_out (int): Number of output filters.
    References
        [1] : "Filterbank design for end-to-end speech separation". ICASSP 2020.
        Manuel Pariente, Samuele Cornell, Antoine Deleforge, Emmanuel Vincent.
    """

    def __init__(
        self, n_filters, kernel_size, stride=None, sample_rate=8000.0, **kwargs
    ):
        super().__init__(n_filters, kernel_size, stride=stride, sample_rate=sample_rate)
        self._filters = nn.Parameter(torch.ones(n_filters, 1, kernel_size))
        for p in self.parameters():
            nn.init.xavier_normal_(p)

    def filters(self):
        return self._filters


free = FreeFB
