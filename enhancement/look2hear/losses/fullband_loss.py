from tokenize import Single
from turtle import forward
from ..models.fullband import stft
import torch.nn as nn
import torch
import numpy as np
from torch.nn.modules.loss import _Loss

EPSILON = np.finfo(np.float32).eps

def istft(features, n_fft, hop_length, win_length, length=None, input_type="complex"):
    """Wrapper of the official torch.istft.

    Args:
        features: [B, F, T] (complex) or ([B, F, T], [B, F, T]) (mag and phase)
        n_fft: num of FFT
        hop_length: hop length
        win_length: hanning window size
        length: expected length of istft
        use_mag_phase: use mag and phase as the input ("features")

    Returns:
        single-channel speech of shape [B, T]
    """
    if input_type == "real_imag":
        # the feature is (real, imag) or [real, imag]
        assert isinstance(features, tuple) or isinstance(features, list)
        real, imag = features
        features = torch.complex(real, imag)
    elif input_type == "complex":
        assert torch.is_complex(features), "The input feature is not complex."
    elif input_type == "mag_phase":
        # the feature is (mag, phase) or [mag, phase]
        assert isinstance(features, tuple) or isinstance(features, list)
        mag, phase = features
        features = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
    else:
        raise NotImplementedError(
            "Only 'real_imag', 'complex', and 'mag_phase' are supported."
        )

    return torch.istft(
        features,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(n_fft, device=features.device),
        length=length,
    )

class SingleSrcNegSDR(_Loss):
    def __init__(
        self, sdr_type, zero_mean=True, take_log=True, reduction="none", EPS=1e-8
    ):
        assert reduction != "sum", NotImplementedError
        super().__init__(reduction=reduction)

        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = 1e-8

    def forward(self, ests, targets):
        if targets.size() != ests.size() or targets.ndim != 2:
            raise TypeError(
                f"Inputs must be of shape [batch, time], got {targets.size()} and {ests.size()} instead"
            )
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=1, keepdim=True)
            mean_estimate = torch.mean(ests, dim=1, keepdim=True)
            targets = targets - mean_source
            ests = ests - mean_estimate
        # Step 2. Pair-wise SI-SDR.
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, 1]
            dot = torch.sum(ests * targets, dim=1, keepdim=True)
            # [batch, 1]
            s_target_energy = torch.sum(targets ** 2, dim=1, keepdim=True) + self.EPS
            # [batch, time]
            scaled_target = dot * targets / s_target_energy
        else:
            # [batch, time]
            scaled_target = targets
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = ests - targets
        else:
            e_noise = ests - scaled_target
        # [batch]
        losses = torch.sum(scaled_target ** 2, dim=1) / (
            torch.sum(e_noise ** 2, dim=1) + self.EPS
        )
        if self.take_log:
            losses = 10 * torch.log10(losses + self.EPS)
        losses = losses.mean() if self.reduction == "mean" else losses
        return -losses

def compress_cIRM(mask, K=10, C=0.1):
    """Compress the value of cIRM from (-inf, +inf) to [-K ~ K].

    References:
        https://ieeexplore.ieee.org/document/7364200
    """
    if torch.is_tensor(mask):
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - torch.exp(-C * mask)) / (1 + torch.exp(-C * mask))
    else:
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - np.exp(-C * mask)) / (1 + np.exp(-C * mask))
    return mask

def decompress_cIRM(mask, K=10, limit=9.9):
    """Decompress cIRM from [-K ~ K] to [-inf, +inf].

    Args:
        mask: cIRM mask
        K: default 10
        limit: default 0.1

    References:
        https://ieeexplore.ieee.org/document/7364200
    """
    mask = (
        limit * (mask >= limit)
        - limit * (mask <= -limit)
        + mask * (torch.abs(mask) < limit)
    )
    mask = -K * torch.log((K - mask) / (K + mask))
    return mask


def build_complex_ideal_ratio_mask(
    noisy_real, noisy_imag, clean_real, clean_imag
) -> torch.Tensor:
    """Build the complex ratio mask.

    Args:
        noisy: [B, F, T], noisy complex-valued stft coefficients
        clean: [B, F, T], clean complex-valued stft coefficients

    References:
        https://ieeexplore.ieee.org/document/7364200

    Returns:
        [B, F, T, 2]
    """
    denominator = torch.square(noisy_real) + torch.square(noisy_imag) + EPSILON

    mask_real = (noisy_real * clean_real + noisy_imag * clean_imag) / denominator
    mask_imag = (noisy_real * clean_imag - noisy_imag * clean_real) / denominator

    complex_ratio_mask = torch.stack((mask_real, mask_imag), dim=-1)

    return compress_cIRM(complex_ratio_mask, K=10, C=0.1)

class FullbandLoss(nn.Module):
    def __init__(self,n_fft, hop_length, win_length,):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.loss_function = nn.MSELoss()
    
    def forward(self, ests, refs):
        """
            ests: [B, 2, F, T]
            refs: [B, T]
        """
        cRM, noisy_real, noisy_imag = ests
        _, _, clean_real, clean_imag = stft(refs, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        cRM = cRM.permute(0, 2, 3, 1)
        cIRM = build_complex_ideal_ratio_mask(noisy_real, noisy_imag, clean_real, clean_imag)  # [B, F, T, 2]
        loss = self.loss_function(cIRM, cRM)
        return loss

class FullbandEval(nn.Module):
    def __init__(self,n_fft, hop_length, win_length,):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.loss_function = SingleSrcNegSDR(sdr_type="sisdr", reduction="mean")
        
    def forward(self, ests, refs):
        """
            ests: [B, 2, F, T]
            refs: [B, T]
        """
        cRM, noisy_real, noisy_imag = ests
        cRM = cRM.permute(0, 2, 3, 1)
        cRM = decompress_cIRM(cRM)
        # import pdb; pdb.set_trace()
        enhanced_real = cRM[..., 0] * noisy_real - cRM[..., 1] * noisy_imag
        enhanced_imag = cRM[..., 1] * noisy_real + cRM[..., 0] * noisy_imag
        enhanced = istft(
            (enhanced_real, enhanced_imag),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            length=refs.size(-1),
            input_type="real_imag",
        )
        loss = self.loss_function(enhanced, refs)
        return loss
    
def inference(ests, n_fft, hop_length, win_length, length):
    cRM, noisy_real, noisy_imag = ests
    cRM = cRM.permute(0, 2, 3, 1)
    cRM = decompress_cIRM(cRM)
    # import pdb; pdb.set_trace()
    enhanced_real = cRM[..., 0] * noisy_real - cRM[..., 1] * noisy_imag
    enhanced_imag = cRM[..., 1] * noisy_real + cRM[..., 0] * noisy_imag
    enhanced = istft(
        (enhanced_real, enhanced_imag),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        length=length,
        input_type="real_imag",
    )
    return enhanced