###
# Author: Kai Li
# Date: 2021-06-22 12:41:36
# LastEditors: Please set LastEditors
# LastEditTime: 2022-06-05 14:48:00
###
import csv
from sympy import im
import torch
import numpy as np
import logging

from torch_mir_eval.separation import bss_eval_sources
import fast_bss_eval
from ..losses import (
    PITLossWrapper,
    pairwise_neg_sisdr,
    pairwise_neg_snr,
    singlesrc_neg_sisdr,
    PairwiseNegSDR,
    SingleSrcNegSDR
)
# from .dnsmos import DNSMOS
from .sigmos import SigMOS
from .asr import ASR
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility

logger = logging.getLogger(__name__)

def is_silent(wav, threshold=1e-4):
    return torch.sum(wav ** 2) / wav.numel() < threshold

class MetricsTracker:
    def __init__(self, save_file: str = ""):
        self.all_sdrs = []
        self.all_sdrs_i = []
        self.all_sisnrs = []
        self.all_sisnrs_i = []
        self.all_pesq_nb = []
        self.all_pesq_wb = []
        self.all_stoi = []
        self.all_mos_col = []
        self.all_mos_disc = []
        self.all_mos_loud = []
        self.all_mos_noise = []
        self.all_mos_reverb = []
        self.all_mos_sig = []
        self.all_mos_ovrl = []
        self.all_asr = []
        self.all_start_idx = []
        self.all_end_idx = []
        
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i", 'pesq_nb', 'pesq_wb', 'stoi', "asr", "MOS_COL", "MOS_DISC", "MOS_LOUD", "MOS_NOISE", "MOS_REVERB", "MOS_SIG", "MOS_OVRL", "start_idx", "end_idx"]
        self.results_csv = open(save_file, "w")
        self.writer = csv.DictWriter(self.results_csv, fieldnames=csv_columns)
        self.writer.writeheader()
        self.pit_sisnr = PITLossWrapper(
            PairwiseNegSDR("snr", zero_mean=False), pit_from="pw_mtx"
        )
        self.pesq_cal_nb = PerceptualEvaluationSpeechQuality(16000, "nb")
        self.pesq_cal_wb = PerceptualEvaluationSpeechQuality(16000, "wb")
        
        self.sigmos_cal = SigMOS("sigmos")
        
        self.asr = ASR()

    def __call__(self, mix, clean, estimate, key, spks_id, start_idx, end_idx):
        sisnr, return_ests = self.pit_sisnr(estimate.unsqueeze(0), clean.unsqueeze(0), return_ests=True)
        return_ests = return_ests.squeeze(0)
        for idx in range(clean.shape[0]):
            if is_silent(clean[idx]):
                continue
            # sisnr
            try:
                if clean[idx].numel() == 0 or return_ests[idx].numel() == 0:
                    continue
                sisnr = fast_bss_eval.si_sdr(clean[idx].unsqueeze(0), return_ests[idx].unsqueeze(0), zero_mean=True)
                sisnr_baseline = fast_bss_eval.si_sdr(clean[idx].unsqueeze(0), mix.unsqueeze(0), zero_mean=True)
                sisnr_i = sisnr - sisnr_baseline
                
                # sdr
                # import pdb; pdb.set_trace()
                sdr = fast_bss_eval.sdr(clean[idx].unsqueeze(0), return_ests[idx].unsqueeze(0), zero_mean=True)
                sdr_baseline = fast_bss_eval.sdr(clean[idx].unsqueeze(0), mix.unsqueeze(0), zero_mean=True)
                sdr_i = sdr - sdr_baseline
                # import pdb; pdb.set_trace()
                
            except:
                try:
                    sisnr = fast_bss_eval.si_sdr(clean[idx].unsqueeze(0), return_ests[idx].unsqueeze(0), zero_mean=True)
                    # sisnr_baseline = fast_bss_eval.si_sdr(clean[idx].unsqueeze(0), mix.unsqueeze(0), zero_mean=True)
                    sisnr_i = sisnr
                    
                    # sdr
                    # import pdb; pdb.set_trace()
                    sdr = fast_bss_eval.sdr(clean[idx].unsqueeze(0), return_ests[idx].unsqueeze(0), zero_mean=True)
                    # sdr_baseline = fast_bss_eval.sdr(clean[idx].unsqueeze(0), mix.unsqueeze(0), zero_mean=True)
                    sdr_i = sdr
                    # import pdb; pdb.set_trace()
                except:
                    return

            # PESQ
            try:
                pesq_nb = self.pesq_cal_nb(return_ests[idx].unsqueeze(0), clean[idx].unsqueeze(0))
                pesq_wb = self.pesq_cal_wb(return_ests[idx].unsqueeze(0), clean[idx].unsqueeze(0))
            except:
                return 
            
            # STOI
            stoi = short_time_objective_intelligibility(return_ests[idx].unsqueeze(0), clean[idx].unsqueeze(0), 16000).float()
            # import pdb; pdb.set_trace()
            
            # Sigmos
            sigmos_dicts = self.sigmos_cal.run(return_ests[idx].cpu().numpy(), 16000)
            
            # ASR
            asr_result = self.asr(return_ests[idx])
            # import pdb; pdb.set_trace()
            
            row = {
                "snt_id": key + f"/s{spks_id[idx]}.wav",
                "sdr": sdr.item(),
                "sdr_i": sdr_i.item(),
                "si-snr": sisnr.item(),
                "si-snr_i": sisnr_i.item(),
                'pesq_nb': pesq_nb.item(),
                'pesq_wb': pesq_wb.item(),
                'stoi': stoi.cpu().item(),
                'asr': asr_result,
                "start_idx": start_idx,
                "end_idx": end_idx
            }
            row.update(sigmos_dicts)
            
            self.writer.writerow(row)
            # Metric Accumulation
            self.all_sdrs.append(sdr.item())
            self.all_sdrs_i.append(sdr_i.item())
            self.all_sisnrs.append(sisnr.item())
            self.all_sisnrs_i.append(sisnr_i.item())
            self.all_pesq_nb.append(pesq_nb.item())
            self.all_pesq_wb.append(pesq_wb.item())
            self.all_stoi.append(stoi.cpu().item())
            self.all_asr.append(asr_result)
            self.all_mos_col.append(sigmos_dicts["MOS_COL"])
            self.all_mos_disc.append(sigmos_dicts["MOS_DISC"])
            self.all_mos_loud.append(sigmos_dicts["MOS_LOUD"])
            self.all_mos_noise.append(sigmos_dicts["MOS_NOISE"])
            self.all_mos_reverb.append(sigmos_dicts["MOS_REVERB"])
            self.all_mos_sig.append(sigmos_dicts["MOS_SIG"])
            self.all_mos_ovrl.append(sigmos_dicts["MOS_OVRL"])
            self.all_start_idx.append(start_idx)
            self.all_end_idx.append(end_idx)
    
    def update(self, ):
        return {"sdr": np.array(self.all_sdrs).mean(),
                "si-snr": np.array(self.all_sisnrs).mean()
                }

    def final(self,):
        row = {
            "snt_id": "avg",
            "sdr": np.array(self.all_sdrs).mean(),
            "sdr_i": np.array(self.all_sdrs_i).mean(),
            "si-snr": np.array(self.all_sisnrs).mean(),
            "si-snr_i": np.array(self.all_sisnrs_i).mean(),
            "pesq_nb": np.array(self.all_pesq_nb).mean(),
            "pesq_wb": np.array(self.all_pesq_wb).mean(),
            "stoi": np.array(self.all_stoi).mean(),
            "asr": None,
            "start_idx": None,
            "end_idx": None,
            "MOS_COL": np.array(self.all_mos_col).mean(),
            "MOS_DISC": np.array(self.all_mos_disc).mean(),
            "MOS_LOUD": np.array(self.all_mos_loud).mean(),
            "MOS_NOISE": np.array(self.all_mos_noise).mean(),
            "MOS_REVERB": np.array(self.all_mos_reverb).mean(),
            "MOS_SIG": np.array(self.all_mos_sig).mean(),
            "MOS_OVRL": np.array(self.all_mos_ovrl).mean(),
        }
        self.writer.writerow(row)
        row = {
            "snt_id": "std",
            "sdr": np.array(self.all_sdrs).std(),
            "sdr_i": np.array(self.all_sdrs_i).std(),
            "si-snr": np.array(self.all_sisnrs).std(),
            "si-snr_i": np.array(self.all_sisnrs_i).std(),
            "pesq_nb": np.array(self.all_pesq_nb).std(),
            "pesq_wb": np.array(self.all_pesq_wb).std(),
            "stoi": np.array(self.all_stoi).std(),
            "asr": None,
            "start_idx": None,
            "end_idx": None,
            "MOS_COL": np.array(self.all_mos_col).std(),
            "MOS_DISC": np.array(self.all_mos_disc).std(),
            "MOS_LOUD": np.array(self.all_mos_loud).std(),
            "MOS_NOISE": np.array(self.all_mos_noise).std(),
            "MOS_REVERB": np.array(self.all_mos_reverb).std(),
            "MOS_SIG": np.array(self.all_mos_sig).std(),
            "MOS_OVRL": np.array(self.all_mos_ovrl).std(),
        }
        self.writer.writerow(row)
        self.results_csv.close()
