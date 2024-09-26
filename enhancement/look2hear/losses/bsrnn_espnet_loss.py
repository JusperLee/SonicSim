import logging
import math
from abc import ABC, abstractmethod

import fast_bss_eval
import torch
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor
is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")
from ..layers import Stft
import torch.nn as nn
from torch.nn.modules.loss import _Loss

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

class AbsEnhLoss(torch.nn.Module, ABC):
    """Base class for all Enhancement loss modules."""

    # the name will be the key that appears in the reporter
    @property
    def name(self) -> str:
        return NotImplementedError

    # This property specifies whether the criterion will only
    # be evaluated during the inference stage
    @property
    def only_for_test(self) -> bool:
        return False

    @abstractmethod
    def forward(
        self,
        ref,
        inf,
    ) -> torch.Tensor:
        # the return tensor should be shape of (batch)
        raise NotImplementedError


class TimeDomainLoss(AbsEnhLoss, ABC):
    """Base class for all time-domain Enhancement loss modules."""

    @property
    def name(self) -> str:
        return self._name

    @property
    def only_for_test(self) -> bool:
        return self._only_for_test

    @property
    def is_noise_loss(self) -> bool:
        return self._is_noise_loss

    @property
    def is_dereverb_loss(self) -> bool:
        return self._is_dereverb_loss

    def __init__(
        self,
        name,
        only_for_test=False,
        is_noise_loss=False,
        is_dereverb_loss=False,
    ):
        super().__init__()
        # only used during validation
        self._only_for_test = only_for_test
        # only used to calculate the noise-related loss
        self._is_noise_loss = is_noise_loss
        # only used to calculate the dereverberation-related loss
        self._is_dereverb_loss = is_dereverb_loss
        if is_noise_loss and is_dereverb_loss:
            raise ValueError(
                "`is_noise_loss` and `is_dereverb_loss` cannot be True at the same time"
            )
        if is_noise_loss and "noise" not in name:
            name = name + "_noise"
        if is_dereverb_loss and "dereverb" not in name:
            name = name + "_dereverb"
        self._name = name


EPS = torch.finfo(torch.get_default_dtype()).eps


class MultiResL1SpecLoss(TimeDomainLoss):
    """Multi-Resolution L1 time-domain + STFT mag loss

    Reference:
    Lu, Y. J., Cornell, S., Chang, X., Zhang, W., Li, C., Ni, Z., ... & Watanabe, S.
    Towards Low-Distortion Multi-Channel Speech Enhancement:
    The ESPNET-Se Submission to the L3DAS22 Challenge. ICASSP 2022 p. 9201-9205.

    Attributes:
        window_sz: (list)
            list of STFT window sizes.
        hop_sz: (list, optional)
            list of hop_sizes, default is each window_sz // 2.
        eps: (float)
            stability epsilon
        time_domain_weight: (float)
            weight for time domain loss.
        normalize_variance (bool)
            whether or not to normalize the variance when calculating the loss.
        reduction (str)
            select from "sum" and "mean"
    """

    def __init__(
        self,
        window_sz=[512],
        hop_sz=None,
        eps=1e-8,
        time_domain_weight=0.5,
        normalize_variance=False,
        reduction="sum",
        name=None,
        only_for_test=False,
        is_noise_loss=False,
        is_dereverb_loss=False,
    ):
        _name = "TD_L1_loss" if name is None else name
        super().__init__(
            _name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )

        assert all([x % 2 == 0 for x in window_sz])
        self.window_sz = window_sz

        if hop_sz is None:
            self.hop_sz = [x // 2 for x in window_sz]
        else:
            self.hop_sz = hop_sz

        self.time_domain_weight = time_domain_weight
        self.normalize_variance = normalize_variance
        self.eps = eps
        self.stft_encoders = torch.nn.ModuleList([])
        for w, h in zip(self.window_sz, self.hop_sz):
            stft_enc = Stft(
                n_fft=w,
                win_length=w,
                hop_length=h,
                window=None,
                center=True,
                normalized=False,
                onesided=True,
            )
            self.stft_encoders.append(stft_enc)

        assert reduction in ("sum", "mean")
        self.reduction = reduction

    @property
    def name(self) -> str:
        return "l1_timedomain+magspec_loss"

    def get_magnitude(self, stft, eps=1e-06):
        if is_torch_1_9_plus:
            stft = torch.complex(stft[..., 0], stft[..., 1])
            return stft.abs()
        else:
            stft = ComplexTensor(stft[..., 0], stft[..., 1])
            return (stft.real.pow(2) + stft.imag.pow(2) + eps).sqrt()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self,
        target: torch.Tensor,
        estimate: torch.Tensor,
    ):
        """forward.

        Args:
            target: (Batch, T)
            estimate: (Batch, T)
        Returns:
            loss: (Batch,)
        """
        assert target.shape == estimate.shape, (target.shape, estimate.shape)
        half_precision = (torch.float16, torch.bfloat16)
        if target.dtype in half_precision or estimate.dtype in half_precision:
            target = target.float()
            estimate = estimate.float()
        if self.normalize_variance:
            target = target / torch.std(target, dim=1, keepdim=True)
            estimate = estimate / torch.std(estimate, dim=1, keepdim=True)
        # shape bsz, samples
        scaling_factor = torch.sum(estimate * target, -1, keepdim=True) / (
            torch.sum(estimate**2, -1, keepdim=True) + self.eps
        )
        if self.reduction == "sum":
            time_domain_loss = torch.sum(
                (estimate * scaling_factor - target).abs(), dim=-1
            )
        elif self.reduction == "mean":
            time_domain_loss = torch.mean(
                (estimate * scaling_factor - target).abs(), dim=-1
            )

        if len(self.stft_encoders) == 0:
            return time_domain_loss
        else:
            spectral_loss = torch.zeros_like(time_domain_loss)
            for stft_enc in self.stft_encoders:
                target_mag = self.get_magnitude(stft_enc(target)[0])
                estimate_mag = self.get_magnitude(
                    stft_enc(estimate * scaling_factor)[0]
                )
                if self.reduction == "sum":
                    c_loss = torch.sum((estimate_mag - target_mag).abs(), dim=(1, 2))
                elif self.reduction == "mean":
                    c_loss = torch.mean((estimate_mag - target_mag).abs(), dim=(1, 2))
                spectral_loss += c_loss

            return time_domain_loss * self.time_domain_weight + (
                1 - self.time_domain_weight
            ) * spectral_loss / len(self.stft_encoders)
            
class BSRNNESPNetLoss(nn.Module):
    def __init__(
        self,
        window_sz=[512],
        hop_sz=None,
        eps=1e-8,
        time_domain_weight=0.5,
        normalize_variance=False,
        reduction="sum",
        name=None,
        only_for_test=False,
        is_noise_loss=False,
        is_dereverb_loss=False,
    ):
        super().__init__()
        self.loss_function = MultiResL1SpecLoss(
            window_sz=window_sz,
            hop_sz=hop_sz,
            eps=eps,
            time_domain_weight=time_domain_weight,
            normalize_variance=normalize_variance,
            reduction=reduction,
            name=name,
            only_for_test=only_for_test,
            is_noise_loss=is_noise_loss,
            is_dereverb_loss=is_dereverb_loss,
        )
        
    def forward(self, ests, targets):
        loss = self.loss_function(targets, ests)
        return loss.mean()

class BSRNNESPNetEval(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.loss_function = SingleSrcNegSDR(sdr_type="sisdr", reduction="mean")

    def forward(self, ests, targets):
        # input: [B, N, T]
        # target: [B, N, T]
        loss = self.loss_function(ests, targets)
        return loss