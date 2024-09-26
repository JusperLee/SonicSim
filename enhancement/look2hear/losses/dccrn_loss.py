from tokenize import Single
from turtle import forward
from ..models.fullband import stft
import torch.nn as nn
import torch
import numpy as np
from torch.nn.modules.loss import _Loss

EPSILON = np.finfo(np.float32).eps


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


class DCCRNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function = SingleSrcNegSDR(sdr_type="sisdr", reduction="mean")
    
    def forward(self, ests, refs):
        """
            ests: [B, 2, F, T]
            refs: [B, T]
        """
        loss = self.loss_function(ests, refs)
        return loss

class DCCRNEval(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function = SingleSrcNegSDR(sdr_type="sisdr", reduction="mean")
        
    def forward(self, ests, refs):
        loss = self.loss_function(ests, refs)
        return loss