import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from ..models.frcrn import ConvSTFT, ConviSTFT

def l2_norm(s1, s2):
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def si_snr(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_noise = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_noise, e_noise)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return torch.mean(snr)

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

class FRCRNLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stft = ConvSTFT(
            640,
            320,
            640,
            "hann",
            feature_type='complex',
            fix=True)
        self.istft = ConviSTFT(
            640,
            320,
            640,
            "hann",
            feature_type='complex',
            fix=True)
        
        self.feat_dim = 640 // 2 + 1

    def forward(self, ests, refs):
        noisy = ests[0]
        out_list = ests[1]
        labels = refs
        count = 0
        while count < len(out_list):
            est_spec = out_list[count]
            count = count + 1
            est_wav = out_list[count]
            count = count + 1
            est_mask = out_list[count]
            count = count + 1
            if count != 3:
                amp_loss, phase_loss, SiSNR_loss = self.loss_1layer(
                    noisy, est_spec, est_wav, labels, est_mask, "Mix")
                loss = amp_loss + phase_loss + SiSNR_loss
        return loss

    def loss_1layer(self, noisy, est, est_wav, labels, cmp_mask, mode='Mix'):
        r""" Compute the loss by mode
        mode == 'Mix'
            est: [B, F*2, T]
            labels: [B, F*2,T]
        mode == 'SiSNR'
            est: [B, T]
            labels: [B, T]
        """
        if mode == 'SiSNR':
            if labels.dim() == 3:
                labels = torch.squeeze(labels, 1)
            if est_wav.dim() == 3:
                est_wav = torch.squeeze(est_wav, 1)
            return -si_snr(est_wav, labels)
        elif mode == 'Mix':

            if labels.dim() == 3:
                labels = torch.squeeze(labels, 1)
            if est_wav.dim() == 3:
                est_wav = torch.squeeze(est_wav, 1)
            SiSNR_loss = -si_snr(est_wav, labels)

            b, d, t = est.size()
            S = self.stft(labels)
            Sr = S[:, :self.feat_dim, :]
            Si = S[:, self.feat_dim:, :]
            Y = self.stft(noisy)
            Yr = Y[:, :self.feat_dim, :]
            Yi = Y[:, self.feat_dim:, :]
            Y_pow = Yr**2 + Yi**2
            gth_mask = torch.cat([(Sr * Yr + Si * Yi) / (Y_pow + 1e-8),
                                  (Si * Yr - Sr * Yi) / (Y_pow + 1e-8)], 1)
            gth_mask[gth_mask > 2] = 1
            gth_mask[gth_mask < -2] = -1
            amp_loss = F.mse_loss(gth_mask[:, :self.feat_dim, :],
                                  cmp_mask[:, :self.feat_dim, :]) * d
            phase_loss = F.mse_loss(gth_mask[:, self.feat_dim:, :],
                                    cmp_mask[:, self.feat_dim:, :]) * d
            return amp_loss, phase_loss, SiSNR_loss
        
class FRCRNEval(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function = SingleSrcNegSDR(sdr_type="sisdr", reduction="mean")
        
    def forward(self, ests, refs):
        ests = ests[1][1]
        loss = self.loss_function(ests, refs)
        return loss