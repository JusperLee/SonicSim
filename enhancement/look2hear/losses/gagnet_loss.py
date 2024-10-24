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

class ComMagEuclideanLoss(object):
    def __init__(self, alpha, l_type):
        self.alpha = alpha
        self.l_type = l_type

    def __call__(self, esti_list, label, frame_list):
        alpha_list = [0.1 for _ in range(len(esti_list))]
        alpha_list[-1] = 1
        mask_for_loss = []
        utt_num = label.size()[0]
        with torch.no_grad():
            for i in range(utt_num):
                tmp_mask = torch.ones((frame_list[i], label.size()[-2]), dtype=label.dtype)
                mask_for_loss.append(tmp_mask)
            mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(label.device)
            mask_for_loss = mask_for_loss.transpose(-2, -1).contiguous()
            com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
        loss1, loss2 = 0., 0.
        mag_label = torch.norm(label, dim=1)
        for i in range(len(esti_list)):
            mag_esti = torch.norm(esti_list[i], dim=1)
            loss1 = loss1 + alpha_list[i] * (((esti_list[i] - label) ** 2.0) * com_mask_for_loss).sum() / com_mask_for_loss.sum()
            loss2 = loss2 + alpha_list[i] * (((mag_esti - mag_label) ** 2.0) * mask_for_loss).sum() / mask_for_loss.sum()
        return 0.5 * (loss1 + loss2)

class GaGNetLoss(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.loss_function = ComMagEuclideanLoss(alpha=0.5, l_type="L2")
    
    def forward(self, ests, refs):
        """
            ests: [B, 2, F, T]
            refs: [B, T]
        """
        c = torch.sqrt(refs.shape[-1] / torch.sum(refs ** 2.0, dim=-1))
        refs = refs * c.unsqueeze(-1)
        batch_target_stft = torch.stft(
            refs,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(refs.device),
            return_complex=True)  # (B,F,T,2)
        batch_target_stft = torch.view_as_real(batch_target_stft)
        batch_frame_list = []
        batch_wav_len_list = [refs.shape[-1] for i in range(refs.shape[0])]
        for i in range(len(batch_wav_len_list)):
            curr_frame_num = (batch_wav_len_list[i] - self.win_length + self.win_length) // self.hop_length + 1
            batch_frame_list.append(curr_frame_num)
        # target
        batch_target_mag, batch_target_phase = torch.norm(batch_target_stft, dim=-1)**0.5, \
                                                torch.atan2(batch_target_stft[..., -1],
                                                            batch_target_stft[..., 0])
        batch_target_stft = torch.stack((batch_target_mag*torch.cos(batch_target_phase),
                                            batch_target_mag*torch.sin(batch_target_phase)), dim=-1)
        batch_target_stft = batch_target_stft.permute(0,3,1,2)
        # stagewise loss
        batch_loss = self.loss_function(ests, batch_target_stft, batch_frame_list)
        
        return batch_loss

class GaGNetEval(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
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
        # est
        ests = ests[-1].permute(0,2,3,1)
        batch_spec_mag, batch_spec_phase = torch.norm(ests, dim=-1)**2.0,\
                                            torch.atan2(ests[..., -1], ests[...,0])
        ests = torch.stack((batch_spec_mag*torch.cos(batch_spec_phase),
                                        batch_spec_mag*torch.sin(batch_spec_phase)), dim=-1)
        # import pdb; pdb.set_trace()
        ests = torch.complex(real=ests[..., 0], imag=ests[..., 1])
        # import pdb; pdb.set_trace()
        batch_est_wav = torch.functional.istft(ests,
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length,
                                    win_length=self.win_length,
                                    window=torch.hann_window(self.win_length).to(refs.device),
                                    length=refs.shape[-1],
                                    return_complex=False)  # (B,L)
        
        loss = self.loss_function(batch_est_wav, refs)
        return loss