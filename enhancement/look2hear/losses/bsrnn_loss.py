import torch.nn as nn
import torch
import numpy as np
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

class BSRNNLoss(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        # self.n_fft = n_fft
        # self.hop_length = hop_length
        # self.win_length = win_length
        self.p = 0.3
        self.loss_function = SingleSrcNegSDR(sdr_type="sisdr", reduction="mean")

    def forward(self, ests, targets):
        # input: [B, N, T]
        # target: [B, N, T]
        loss = 0.0
        I = 4
        for win in [10, 20, 30, 40]:
            n_fft = int(win * 16)
            hop_length = n_fft // 2
            est_spec = torch.stft(ests.view(-1, ests.shape[-1]), n_fft=n_fft, hop_length=hop_length, 
                            window=torch.hann_window(n_fft).to(ests.device).float(),
                            return_complex=True)
            est_target = torch.stft(targets.view(-1, targets.shape[-1]), n_fft=n_fft, hop_length=hop_length, 
                            window=torch.hann_window(n_fft).to(targets.device).float(),
                                    return_complex=True)
            loss = loss + (est_spec.abs() - est_target.abs()).abs().mean() / (est_target.abs().mean() + torch.finfo(torch.float32).eps)
            # all_freq_loss.append(freq_loss)
        # freq_loss = sum(all_freq_loss) / len(all_freq_loss)
        return loss/I
        # wav_loss = (ests - targets).abs().mean()
        # return freq_loss + wav_loss
        # loss = self.loss_function(ests, targets)
        # return loss
    
class BSRNNEval(nn.Module):
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