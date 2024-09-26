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

logger = logging.getLogger(__name__)

def is_silent(wav, threshold=1e-4):
    return torch.sum(wav ** 2) / wav.numel() < threshold

class MetricsTrackerNoASR:
    def __init__(self, save_file: str = ""):
        self.all_sdrs = []
        self.all_sdrs_i = []
        self.all_sisnrs = []
        self.all_sisnrs_i = []
        
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]
        self.results_csv = open(save_file, "w")
        self.writer = csv.DictWriter(self.results_csv, fieldnames=csv_columns)
        self.writer.writeheader()
        self.pit_sisnr = PITLossWrapper(
            PairwiseNegSDR("snr", zero_mean=False), pit_from="pw_mtx"
        )

    def __call__(self, mix, clean, estimate, key):
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

            
            row = {
                "snt_id": key,
                "sdr": sdr.item(),
                "sdr_i": sdr_i.item(),
                "si-snr": sisnr.item(),
                "si-snr_i": sisnr_i.item()
            }
            
            self.writer.writerow(row)
            # Metric Accumulation
            self.all_sdrs.append(sdr.item())
            self.all_sdrs_i.append(sdr_i.item())
            self.all_sisnrs.append(sisnr.item())
            self.all_sisnrs_i.append(sisnr_i.item())
    
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
        }
        self.writer.writerow(row)
        row = {
            "snt_id": "std",
            "sdr": np.array(self.all_sdrs).std(),
            "sdr_i": np.array(self.all_sdrs_i).std(),
            "si-snr": np.array(self.all_sisnrs).std(),
            "si-snr_i": np.array(self.all_sisnrs_i).std()
        }
        self.writer.writerow(row)
        self.results_csv.close()
