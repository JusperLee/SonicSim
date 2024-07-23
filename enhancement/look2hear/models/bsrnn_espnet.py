from itertools import accumulate

import torch
import torch.nn as nn
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor
is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")

from ..layers import Stft

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

from torch_complex import functional as FC
from .base_model import BaseModel

EPS = torch.finfo(torch.double).eps

class AbsEncoder(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError

    def forward_streaming(self, input: torch.Tensor):
        raise NotImplementedError

    def streaming_frame(self, audio: torch.Tensor):
        """streaming_frame. It splits the continuous audio into frame-level
        audio chunks in the streaming *simulation*. It is noted that this
        function takes the entire long audio as input for a streaming simulation.
        You may refer to this function to manage your streaming input
        buffer in a real streaming application.

        Args:
            audio: (B, T)
        Returns:
            chunked: List [(B, frame_size),]
        """
        NotImplementedError

class STFTEncoder(AbsEncoder):
    """STFT encoder for speech enhancement and separation"""

    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window="hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        use_builtin_complex: bool = True,
        default_fs: int = 16000,
        spec_transform_type: str = None,
        spec_factor: float = 0.15,
        spec_abs_exponent: float = 0.5,
    ):
        super().__init__()
        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )

        self._output_dim = n_fft // 2 + 1 if onesided else n_fft
        self.use_builtin_complex = use_builtin_complex
        self.win_length = win_length if win_length else n_fft
        self.hop_length = hop_length
        self.window = window
        self.n_fft = n_fft
        self.center = center
        self.default_fs = default_fs

        # spec transform related. See equation (1) in paper
        # 'Speech Enhancement and Dereverberation With Diffusion-Based Generative
        # Models'. The default value of 0.15, 0.5 also come from the paper.
        # spec_transform_type: "exponent", "log", or "none"
        self.spec_transform_type = spec_transform_type
        # the output specturm will be scaled with: spec * self.spec_factor
        self.spec_factor = spec_factor
        # the exponent factor used in the "exponent" transform
        self.spec_abs_exponent = spec_abs_exponent

    def spec_transform_func(self, spec):
        if self.spec_transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1,
                # otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = self.spec_abs_exponent
                spec = spec.abs() ** e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.spec_transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.spec_transform_type == "none":
            spec = spec
        return spec

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input: torch.Tensor, ilens: torch.Tensor, fs: int = None):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]
            fs (int): sampling rate in Hz
                If not None, reconfigure STFT window and hop lengths for a new
                sampling rate while keeping their duration fixed.
        Returns:
            spectrum (ComplexTensor): [Batch, T, (C,) F]
            flens (torch.Tensor): [Batch]
        """
        if fs is not None:
            self._reconfig_for_fs(fs)
        # for supporting half-precision training
        if input.dtype in (torch.float16, torch.bfloat16):
            spectrum, flens = self.stft(input.float(), ilens)
            spectrum = spectrum.to(dtype=input.dtype)
        else:
            spectrum, flens = self.stft(input, ilens)
        if is_torch_1_9_plus and self.use_builtin_complex:
            spectrum = torch.complex(spectrum[..., 0], spectrum[..., 1])
        else:
            spectrum = ComplexTensor(spectrum[..., 0], spectrum[..., 1])

        self._reset_config()

        spectrum = self.spec_transform_func(spectrum)

        return spectrum, flens

    def _reset_config(self):
        """Reset the configuration of STFT window and hop lengths."""
        self._reconfig_for_fs(self.default_fs)

    def _reconfig_for_fs(self, fs):
        """Reconfigure STFT window and hop lengths for a new sampling rate
        while keeping their duration fixed.

        Args:
            fs (int): new sampling rate
        """  # noqa: H405
        self.stft.n_fft = self.n_fft * fs // self.default_fs
        self.stft.win_length = self.win_length * fs // self.default_fs
        self.stft.hop_length = self.hop_length * fs // self.default_fs

    def _apply_window_func(self, input):
        B = input.shape[0]

        window_func = getattr(torch, f"{self.window}_window")
        window = window_func(self.win_length, dtype=input.dtype, device=input.device)
        n_pad_left = (self.n_fft - window.shape[0]) // 2
        n_pad_right = self.n_fft - window.shape[0] - n_pad_left

        windowed = input * window

        windowed = torch.cat(
            [torch.zeros(B, n_pad_left), windowed, torch.zeros(B, n_pad_right)], 1
        )
        return windowed

    def forward_streaming(self, input: torch.Tensor):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, frame_length]
        Return:
            B, 1, F
        """

        assert (
            input.dim() == 2
        ), "forward_streaming only support for single-channel input currently."

        windowed = self._apply_window_func(input)

        feature = (
            torch.fft.rfft(windowed) if self.stft.onesided else torch.fft.fft(windowed)
        )
        feature = feature.unsqueeze(1)
        if not (is_torch_1_9_plus and self.use_builtin_complex):
            feature = ComplexTensor(feature.real, feature.imag)

        feature = self.spec_transform_func(feature)

        return feature

    def streaming_frame(self, audio):
        """streaming_frame. It splits the continuous audio into frame-level
        audio chunks in the streaming *simulation*. It is noted that this
        function takes the entire long audio as input for a streaming simulation.
        You may refer to this function to manage your streaming input
        buffer in a real streaming application.

        Args:
            audio: (B, T)
        Returns:
            chunked: List [(B, frame_size),]
        """  # noqa: H405

        if self.center:
            pad_len = int(self.win_length // 2)
            signal_dim = audio.dim()
            extended_shape = [1] * (3 - signal_dim) + list(audio.size())
            # the default STFT pad mode is "reflect",
            # which is not configurable in STFT encoder,
            # so, here we just use "reflect mode"
            audio = torch.nn.functional.pad(
                audio.view(extended_shape), [pad_len, pad_len], "reflect"
            )
            audio = audio.view(audio.shape[-signal_dim:])

        _, audio_len = audio.shape

        n_frames = 1 + (audio_len - self.win_length) // self.hop_length
        strides = list(audio.stride())

        shape = list(audio.shape[:-1]) + [self.win_length, n_frames]
        strides = strides + [self.hop_length]

        return audio.as_strided(shape, strides, storage_offset=0).unbind(dim=-1)

class AbsDecoder(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward_streaming(self, input_frame: torch.Tensor):
        raise NotImplementedError

    def streaming_merge(self, chunks: torch.Tensor, ilens: torch.tensor = None):
        """streaming_merge. It merges the frame-level processed audio chunks
        in the streaming *simulation*. It is noted that, in real applications,
        the processed audio should be sent to the output channel frame by frame.
        You may refer to this function to manage your streaming output buffer.

        Args:
            chunks: List [(B, frame_size),]
            ilens: [B]
        Returns:
            merge_audio: [B, T]
        """

        raise NotImplementedError

class STFTDecoder(AbsDecoder):
    """STFT decoder for speech enhancement and separation"""

    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window="hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        default_fs: int = 16000,
        spec_transform_type: str = None,
        spec_factor: float = 0.15,
        spec_abs_exponent: float = 0.5,
    ):
        super().__init__()
        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )

        self.win_length = win_length if win_length else n_fft
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.center = center
        self.default_fs = default_fs

        # spec transform related. See equation (1) in paper
        # 'Speech Enhancement and Dereverberation With Diffusion-Based Generative
        # Models'. The default value of 0.15, 0.5 also come from the paper.
        # spec_transform_type: "exponent", "log", or "none"
        self.spec_transform_type = spec_transform_type
        # the output specturm will be scaled with: spec * self.spec_factor
        self.spec_factor = spec_factor
        # the exponent factor used in the "exponent" transform
        self.spec_abs_exponent = spec_abs_exponent

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input: ComplexTensor, ilens: torch.Tensor, fs: int = None):
        """Forward.

        Args:
            input (ComplexTensor): spectrum [Batch, T, (C,) F]
            ilens (torch.Tensor): input lengths [Batch]
            fs (int): sampling rate in Hz
                If not None, reconfigure iSTFT window and hop lengths for a new
                sampling rate while keeping their duration fixed.
        """
        if not isinstance(input, ComplexTensor) and (
            is_torch_1_9_plus and not torch.is_complex(input)
        ):
            raise TypeError("Only support complex tensors for stft decoder")
        if fs is not None:
            self._reconfig_for_fs(fs)

        input = self.spec_back(input)

        bs = input.size(0)
        if input.dim() == 4:
            multi_channel = True
            # input: (Batch, T, C, F) -> (Batch * C, T, F)
            input = input.transpose(1, 2).reshape(-1, input.size(1), input.size(3))
        else:
            multi_channel = False

        # for supporting half-precision training
        if input.dtype in (torch.float16, torch.bfloat16):
            wav, wav_lens = self.stft.inverse(input.float(), ilens)
            wav = wav.to(dtype=input.dtype)
        elif (
            is_torch_complex_tensor(input)
            and hasattr(torch, "complex32")
            and input.dtype == torch.complex32
        ):
            wav, wav_lens = self.stft.inverse(input.cfloat(), ilens)
            wav = wav.to(dtype=input.dtype)
        else:
            wav, wav_lens = self.stft.inverse(input, ilens)

        if multi_channel:
            # wav: (Batch * C, Nsamples) -> (Batch, Nsamples, C)
            wav = wav.reshape(bs, -1, wav.size(1)).transpose(1, 2)

        self._reset_config()
        return wav, wav_lens

    def _reset_config(self):
        """Reset the configuration of iSTFT window and hop lengths."""
        self._reconfig_for_fs(self.default_fs)

    def _reconfig_for_fs(self, fs):
        """Reconfigure iSTFT window and hop lengths for a new sampling rate

        while keeping their duration fixed.

        Args:
            fs (int): new sampling rate
        """
        self.stft.n_fft = self.n_fft * fs // self.default_fs
        self.stft.win_length = self.win_length * fs // self.default_fs
        self.stft.hop_length = self.hop_length * fs // self.default_fs

    def _get_window_func(self):
        window_func = getattr(torch, f"{self.window}_window")
        window = window_func(self.win_length)
        n_pad_left = (self.n_fft - window.shape[0]) // 2
        n_pad_right = self.n_fft - window.shape[0] - n_pad_left  # noqa
        return window

    def spec_back(self, spec):
        if self.spec_transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs() ** (1 / e) * torch.exp(1j * spec.angle())
        elif self.spec_transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.spec_transform_type == "none":
            spec = spec
        return spec

    def forward_streaming(self, input_frame: torch.Tensor):
        """Forward.

        Args:
            input (ComplexTensor): spectrum [Batch, 1, F]
            output: wavs [Batch, 1, self.win_length]
        """
        input_frame = self.spec_back(input_frame)
        input_frame = input_frame.real + 1j * input_frame.imag
        output_wav = (
            torch.fft.irfft(input_frame)
            if self.stft.onesided
            else torch.fft.ifft(input_frame).real
        )

        output_wav = output_wav.squeeze(1)

        n_pad_left = (self.n_fft - self.win_length) // 2
        output_wav = output_wav[..., n_pad_left : n_pad_left + self.win_length]

        return output_wav * self._get_window_func()

    def streaming_merge(self, chunks, ilens=None):
        """streaming_merge. It merges the frame-level processed audio chunks
        in the streaming *simulation*. It is noted that, in real applications,
        the processed audio should be sent to the output channel frame by frame.
        You may refer to this function to manage your streaming output buffer.

        Args:
            chunks: List [(B, frame_size),]
            ilens: [B]
        Returns:
            merge_audio: [B, T]
        """  # noqa: H405

        frame_size = self.win_length
        hop_size = self.hop_length

        num_chunks = len(chunks)
        batch_size = chunks[0].shape[0]
        audio_len = int(hop_size * num_chunks + frame_size - hop_size)

        output = torch.zeros((batch_size, audio_len), dtype=chunks[0].dtype).to(
            chunks[0].device
        )

        for i, chunk in enumerate(chunks):
            output[:, i * hop_size : i * hop_size + frame_size] += chunk

        window_sq = self._get_window_func().pow(2)
        window_envelop = torch.zeros((batch_size, audio_len), dtype=chunks[0].dtype).to(
            chunks[0].device
        )
        for i in range(len(chunks)):
            window_envelop[:, i * hop_size : i * hop_size + frame_size] += window_sq
        output = output / window_envelop

        # We need to trim the front padding away if center.
        start = (frame_size // 2) if self.center else 0
        end = -(frame_size // 2) if ilens.max() is None else start + ilens.max()

        return output[..., start:end]

def new_complex_like(
    ref: Union[torch.Tensor, ComplexTensor],
    real_imag: Tuple[torch.Tensor, torch.Tensor],
):
    if isinstance(ref, ComplexTensor):
        return ComplexTensor(*real_imag)
    elif is_torch_complex_tensor(ref):
        return torch.complex(*real_imag)
    else:
        raise ValueError(
            "Please update your PyTorch version to 1.9+ for complex support."
        )


def is_torch_complex_tensor(c):
    return not isinstance(c, ComplexTensor) and torch.is_complex(c)


def is_complex(c):
    return isinstance(c, ComplexTensor) or is_torch_complex_tensor(c)


def to_complex(c):
    # Convert to torch native complex
    if isinstance(c, ComplexTensor):
        c = c.real + 1j * c.imag
        return c
    elif torch.is_complex(c):
        return c
    else:
        return torch.view_as_complex(c)

class AbsSeparator(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor, OrderedDict]:
        raise NotImplementedError

    def forward_streaming(
        self,
        input_frame: torch.Tensor,
        buffer=None,
    ):
        raise NotImplementedError

    @property
    @abstractmethod
    def num_spk(self):
        raise NotImplementedError

class BSRNN(nn.Module):
    # ported from https://github.com/sungwon23/BSRNN
    def __init__(
        self, input_dim=481, num_channel=16, num_layer=6, target_fs=48000, causal=True
    ):
        """Band-Split RNN (BSRNN).

        References:
            [1] J. Yu, H. Chen, Y. Luo, R. Gu, and C. Weng, “High fidelity speech
            enhancement with band-split RNN,” in Proc. ISCA Interspeech, 2023.
            https://isca-speech.org/archive/interspeech_2023/yu23b_interspeech.html
            [2] J. Yu, and Y. Luo, “Efficient monaural speech enhancement with
            universal sample rate band-split RNN,” in Proc. ICASSP, 2023.
            https://ieeexplore.ieee.org/document/10096020

        Args:
            input_dim (int): maximum number of frequency bins corresponding to
                `target_fs`
            num_channel (int): embedding dimension of each time-frequency bin
            num_layer (int): number of time and frequency RNN layers
            target_fs (int): maximum sampling frequency supported by the model
            causal (bool): Whether or not to adopt causal processing
                if True, LSTM will be used instead of BLSTM for time modeling
        """
        super().__init__()
        self.num_layer = num_layer
        self.band_split = BandSplit(
            input_dim, target_fs=target_fs, channels=num_channel
        )
        self.target_fs = target_fs
        self.causal = causal

        self.norm_time = nn.ModuleList()
        self.rnn_time = nn.ModuleList()
        self.fc_time = nn.ModuleList()
        self.norm_freq = nn.ModuleList()
        self.rnn_freq = nn.ModuleList()
        self.fc_freq = nn.ModuleList()
        hdim = 2 * num_channel
        for i in range(self.num_layer):
            self.norm_time.append(nn.GroupNorm(1, num_channel))
            self.rnn_time.append(
                nn.LSTM(
                    num_channel,
                    hdim,
                    batch_first=True,
                    bidirectional=not causal,
                )
            )
            self.fc_time.append(nn.Linear(hdim if causal else hdim * 2, num_channel))
            self.norm_freq.append(nn.GroupNorm(1, num_channel))
            self.rnn_freq.append(
                nn.LSTM(num_channel, hdim, batch_first=True, bidirectional=True)
            )
            self.fc_freq.append(nn.Linear(4 * num_channel, num_channel))

        self.mask_decoder = MaskDecoder(
            input_dim, self.band_split.subbands, channels=num_channel
        )

    def forward(self, x, fs=None):
        """BSRNN forward.

        Args:
            x (torch.Tensor): input tensor of shape (B, T, F, 2)
            fs (int, optional): sampling rate of the input signal.
                if not None, the input signal will be truncated to only process the
                effective frequency subbands.
                if None, the input signal is assumed to be already truncated to only
                contain effective frequency subbands.
        Returns:
            out (torch.Tensor): output tensor of shape (B, T, F, 2)
        """
        z = self.band_split(x, fs=fs)
        B, N, T, K = z.shape
        skip = z
        for i in range(self.num_layer):
            out = self.norm_time[i](skip)
            out = out.transpose(1, 3).reshape(B * K, T, N)
            out, _ = self.rnn_time[i](out)
            out = self.fc_time[i](out)
            out = out.reshape(B, K, T, N).transpose(1, 3)
            skip = skip + out

            out = self.norm_freq[i](skip)
            out = out.permute(0, 2, 3, 1).contiguous().reshape(B * T, K, N)
            out, _ = self.rnn_freq[i](out)
            out = self.fc_freq[i](out)
            out = out.reshape(B, T, K, N).permute(0, 3, 1, 2).contiguous()
            skip = skip + out

        m, r = self.mask_decoder(skip)
        m = torch.view_as_complex(m)
        r = torch.view_as_complex(r)
        x = torch.view_as_complex(x)
        m = m[..., : x.size(-1)]
        r = r[..., : x.size(-1)]
        return torch.view_as_real(m * x + r)


class BandSplit(nn.Module):
    def __init__(self, input_dim, target_fs=48000, channels=128):
        super().__init__()
        assert input_dim % 2 == 1, input_dim
        n_fft = (input_dim - 1) * 2
        # freq resolution = target_fs / n_fft = freqs[1] - freqs[0]
        freqs = torch.fft.rfftfreq(n_fft, 1.0 / target_fs)
        if input_dim == 481 and target_fs == 48000:
            # n_fft=960 (20ms)
            # first 20 200Hz subbands: [0-200], (200-400], (400-600], ..., (3800-4000]
            # subsequent 6 500Hz subbands: (4000, 4500], ..., (6500, 7000]
            # subsequent 7 2kHz subbands: (7000, 9000], ..., (19000, 21000]
            # final 3kHz subband: (21000, 24000]
            self.subbands = tuple([5] + [4] * 19 + [10] * 6 + [40] * 7 + [60])
        elif input_dim == 161 and target_fs == 16000:
            self.subbands = tuple([2] * 20 + [5] * 6 + [20] * 3 + [31])
        else:
            raise NotImplementedError(
                f"Please define your own subbands for input_dim={input_dim} and "
                f"target_fs={target_fs}"
            )
        self.subband_freqs = freqs[[idx - 1 for idx in accumulate(self.subbands)]]

        self.norm = nn.ModuleList()
        self.fc = nn.ModuleList()
        for i in range(len(self.subbands)):
            self.norm.append(nn.GroupNorm(1, int(self.subbands[i] * 2)))
            self.fc.append(nn.Conv1d(int(self.subbands[i] * 2), channels, 1))

    def forward(self, x, fs=None):
        """BandSplit forward.

        Args:
            x (torch.Tensor): input tensor of shape (B, T, F, 2)
            fs (int, optional): sampling rate of the input signal.
                if not None, the input signal will be truncated to only process the
                effective frequency subbands.
                if None, the input signal is assumed to be already truncated to only
                contain effective frequency subbands.
        Returns:
            z (torch.Tensor): output tensor of shape (B, N, T, K')
                K' might be smaller than len(self.subbands) if fs < self.target_fs.
        """
        if fs is not None:
            assert x.size(2) == sum(self.subbands), (x.size(2), sum(self.subbands))
        hz_band = 0
        for i, subband in enumerate(self.subbands):
            if fs is not None and self.subband_freqs[i] > fs / 2:
                break
            if fs is None and hz_band >= x.size(2):
                break
            x_band = x[:, :, hz_band : hz_band + int(subband), :]
            if int(subband) > x_band.size(2):
                x_band = nn.functional.pad(
                    x_band, (0, 0, 0, int(subband) - x_band.size(2))
                )
            x_band = x_band.reshape(x_band.size(0), x_band.size(1), -1)
            # import pdb; pdb.set_trace()
            out = self.norm[i](x_band.transpose(1, 2))
            # (B, band * 2, T) -> (B, N, T)
            out = self.fc[i](out)

            if i == 0:
                z = out.unsqueeze(-1)
            else:
                z = torch.cat((z, out.unsqueeze(-1)), dim=-1)
            hz_band = hz_band + int(subband)
        return z


class MaskDecoder(nn.Module):
    def __init__(self, freq_dim, subbands, channels=128):
        super().__init__()
        assert freq_dim == sum(subbands), (freq_dim, subbands)
        self.subbands = subbands
        self.freq_dim = freq_dim
        self.mlp_mask = nn.ModuleList()
        self.mlp_residual = nn.ModuleList()
        for subband in self.subbands:
            self.mlp_mask.append(
                nn.Sequential(
                    nn.GroupNorm(1, channels),
                    nn.Conv1d(channels, 4 * channels, 1),
                    nn.Tanh(),
                    nn.Conv1d(4 * channels, int(subband * 4), 1),
                    nn.GLU(dim=1),
                )
            )
            self.mlp_residual.append(
                nn.Sequential(
                    nn.GroupNorm(1, channels),
                    nn.Conv1d(channels, 4 * channels, 1),
                    nn.Tanh(),
                    nn.Conv1d(4 * channels, int(subband * 4), 1),
                    nn.GLU(dim=1),
                )
            )

    def forward(self, x):
        """MaskDecoder forward.

        Args:
            x (torch.Tensor): input tensor of shape (B, N, T, K)
        Returns:
            m (torch.Tensor): output mask of shape (B, T, F, 2)
            r (torch.Tensor): output residual of shape (B, T, F, 2)
        """
        for i in range(len(self.subbands)):
            if i >= x.size(-1):
                break
            x_band = x[:, :, :, i]
            out = self.mlp_mask[i](x_band).transpose(1, 2).contiguous()
            out = out.reshape(out.size(0), out.size(1), -1, 2)
            if i == 0:
                m = out
            else:
                m = torch.cat((m, out), dim=2)

            res = self.mlp_residual[i](x_band).transpose(1, 2).contiguous()
            res = res.reshape(res.size(0), res.size(1), -1, 2)
            if i == 0:
                r = res
            else:
                r = torch.cat((r, res), dim=2)
        # Pad zeros in addition to efffective subbands to cover the full frequency range
        m = nn.functional.pad(m, (0, 0, 0, int(self.freq_dim - m.size(-2))))
        r = nn.functional.pad(r, (0, 0, 0, int(self.freq_dim - r.size(-2))))
        return m, r

class BSRNNSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 1,
        num_channels: int = 16,
        num_layers: int = 6,
        target_fs: int = 48000,
        causal: bool = True,
        ref_channel: Optional[int] = None,
    ):
        """Band-split RNN (BSRNN) separator.

        Reference:
            [1] J. Yu, H. Chen, Y. Luo, R. Gu, and C. Weng, “High fidelity speech
            enhancement with band-split RNN,” in Proc. ISCA Interspeech, 2023.
            https://isca-speech.org/archive/interspeech_2023/yu23b_interspeech.html
            [2] J. Yu, and Y. Luo, “Efficient monaural speech enhancement with
            universal sample rate band-split RNN,” in Proc. ICASSP, 2023.
            https://ieeexplore.ieee.org/document/10096020

        Args:
            input_dim: (int) maximum number of frequency bins corresponding to
                `target_fs`
            num_spk: (int) number of speakers.
            num_channels: (int) feature dimension in the BandSplit block.
            num_layers: (int) number of processing layers.
            target_fs: (int) max sampling frequency that the model can handle.
            causal (bool): whether or not to apply causal modeling.
                if True, LSTM will be used instead of BLSTM for time modeling
            ref_channel: (int) reference channel. not used for now.
        """
        super().__init__()

        self._num_spk = num_spk
        self.ref_channel = ref_channel

        assert num_spk == 1, num_spk
        self.bsrnn = BSRNN(
            input_dim=input_dim,
            num_channel=num_channels,
            num_layer=num_layers,
            target_fs=target_fs,
            causal=causal,
        )

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """BSRNN Forward.

        Args:
            input (torch.Tensor or ComplexTensor): STFT spectrum [B, T, (C,) F (,2)]
            ilens (torch.Tensor): input lengths [Batch]
            additional (Dict or None): other data included in model.
                unused in this model.

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, F), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """
        # B, T, (C,) F, 2
        if is_complex(input):
            feature = torch.stack([input.real, input.imag], dim=-1)
        else:
            assert input.size(-1) == 2, input.shape
            feature = input

        masked = self.bsrnn(feature).unsqueeze(1)
        # B, num_spk, T, F
        if not is_complex(input):
            masked = list(ComplexTensor(masked[..., 0], masked[..., 1]).unbind(1))
        else:
            masked = list(
                new_complex_like(input, (masked[..., 0], masked[..., 1])).unbind(1)
            )

        others = {}
        return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk

class BSRNNESPNet(BaseModel):
    def __init__(
        self,
        n_fft = 960,
        hop_length = 480,
        use_builtin_complex = True,
        num_spk = 1,
        num_channels = 256,  # embedding dimension of each T-F bin
        num_layers = 12,     # number of time and frequency modeling layers
        target_fs = 48000,   # highest sampling frequency to model
        ref_channel = 0,     # used for multichannel signals
        causal = False,      # non-causal
    ):
        super().__init__(sample_rate=16000)
        self.enc = STFTEncoder(
            n_fft, n_fft, hop_length, default_fs=16000, window="hann", use_builtin_complex=use_builtin_complex
        )
        self.dec = STFTDecoder(n_fft, n_fft, hop_length, default_fs=16000, window="hann")
        
        self.separator = BSRNNSeparator(
            n_fft // 2 + 1,
            num_spk=num_spk,
            num_channels=num_channels,
            num_layers=num_layers,
            target_fs=target_fs,
            causal=causal,
            ref_channel=ref_channel,
        )
        
    def forward(self, input: torch.Tensor):
        # input shape: (B, C, T)
        n_samples = input.shape[1]
        ilens = torch.ones(input.shape[0], dtype=torch.long, device=input.device) * n_samples
        feature_mix, flens = self.enc(input, ilens, fs=16000)
        # import pdb; pdb.set_trace()
        feature_pre, flens, others = self.separator(feature_mix, flens)
        
        speech_pre = [self.dec(ps, ilens, fs=16000)[0] for ps in feature_pre]
        return torch.stack(speech_pre, dim=1).view(input.shape[0], -1)
    
    def get_model_args(self):
        model_args = {"n_sample_rate": 2}
        return model_args
        