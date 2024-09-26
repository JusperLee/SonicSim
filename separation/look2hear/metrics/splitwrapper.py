import csv
import torch
import numpy as np
import logging

# from torch_mir_eval.separation import bss_eval_sources
from ..losses import (
    PITLossWrapper,
    pairwise_neg_sisdr,
    pairwise_neg_snr,
    singlesrc_neg_sisdr,
)

logger = logging.getLogger(__name__)


class SPlitMetricsTracker:
    def __init__(self, save_file: str = ""):
        self.one_all_snrs = []
        self.one_all_snrs_i = []
        self.one_all_sisnrs = []
        self.one_all_sisnrs_i = []
        self.two_all_snrs = []
        self.two_all_snrs_i = []
        self.two_all_sisnrs = []
        self.two_all_sisnrs_i = []
        csv_columns = [
            "snt_id",
            "one_snr",
            "one_snr_i",
            "one_si-snr",
            "one_si-snr_i",
            "two_snr",
            "two_snr_i",
            "two_si-snr",
            "two_si-snr_i",
        ]
        self.results_csv = open(save_file, "w")
        self.writer = csv.DictWriter(self.results_csv, fieldnames=csv_columns)
        self.writer.writeheader()
        self.pit_sisnr = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
        self.pit_snr = PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx")

    def __call__(self, mix, clean, estimate, key):
        _, ests_np = self.pit_snr(
            estimate.unsqueeze(0), clean.unsqueeze(0), return_ests=True
        )
        # sisnr
        two_sisnr = self.pit_sisnr(ests_np[:, 0:2], clean.unsqueeze(0)[:, 0:2])
        one_sisnr = self.pit_sisnr(
            ests_np[:, 2].unsqueeze(1), clean.unsqueeze(0)[:, 2].unsqueeze(1)
        )
        mix = torch.stack([mix] * clean.shape[0], dim=0)
        two_sisnr_baseline = self.pit_sisnr(
            mix.unsqueeze(0)[:, 0:2], clean.unsqueeze(0)[:, 0:2]
        )
        one_sisnr_baseline = self.pit_sisnr(
            mix.unsqueeze(0)[:, 2].unsqueeze(1), clean.unsqueeze(0)[:, 2].unsqueeze(1)
        )
        two_sisnr_i = two_sisnr - two_sisnr_baseline
        one_sisnr_i = one_sisnr - one_sisnr_baseline
        # sdr
        two_snr = self.pit_snr(ests_np[:, 0:2], clean.unsqueeze(0)[:, 0:2])
        one_snr = self.pit_snr(
            ests_np[:, 2].unsqueeze(1), clean.unsqueeze(0)[:, 2].unsqueeze(1)
        )
        two_snr_baseline = self.pit_snr(
            mix.unsqueeze(0)[:, 0:2], clean.unsqueeze(0)[:, 0:2]
        )
        one_snr_baseline = self.pit_snr(
            mix.unsqueeze(0)[:, 2].unsqueeze(1), clean.unsqueeze(0)[:, 2].unsqueeze(1)
        )
        two_snr_i = two_snr - two_snr_baseline
        one_snr_i = one_snr - one_snr_baseline

        row = {
            "snt_id": key,
            "one_snr": -one_snr.item(),
            "one_snr_i": -one_snr_i.item(),
            "one_si-snr": -one_sisnr.item(),
            "one_si-snr_i": -one_sisnr_i.item(),
            "two_snr": -two_snr.item(),
            "two_snr_i": -two_snr_i.item(),
            "two_si-snr": -two_sisnr.item(),
            "two_si-snr_i": -two_sisnr_i.item(),
        }
        self.writer.writerow(row)
        # Metric Accumulation
        self.one_all_snrs.append(-one_snr.item())
        self.one_all_snrs_i.append(-one_snr_i.item())
        self.one_all_sisnrs.append(-one_sisnr.item())
        self.one_all_sisnrs_i.append(-one_sisnr_i.item())
        self.two_all_snrs.append(-two_snr.item())
        self.two_all_snrs_i.append(-two_snr_i.item())
        self.two_all_sisnrs.append(-two_sisnr.item())
        self.two_all_sisnrs_i.append(-two_sisnr_i.item())

    def final(self,):
        row = {
            "snt_id": "avg",
            "one_snr": np.array(self.one_all_snrs).mean(),
            "one_snr_i": np.array(self.one_all_snrs_i).mean(),
            "one_si-snr": np.array(self.one_all_sisnrs).mean(),
            "one_si-snr_i": np.array(self.one_all_sisnrs_i).mean(),
            "two_snr": np.array(self.two_all_snrs).mean(),
            "two_snr_i": np.array(self.two_all_snrs_i).mean(),
            "two_si-snr": np.array(self.two_all_sisnrs).mean(),
            "two_si-snr_i": np.array(self.two_all_sisnrs_i).mean(),
        }
        self.writer.writerow(row)
        # logger.info("Mean SISNR is {}".format(row["si-snr"]))
        # logger.info("Mean SISNRi is {}".format(row["si-snr_i"]))
        # logger.info("Mean SDR is {}".format(row["sdr"]))
        # logger.info("Mean SDRi is {}".format(row["sdr_i"]))
        self.results_csv.close()
