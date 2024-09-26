import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import warnings
warnings.filterwarnings("ignore")

import torchaudio

from rich import print
from typing import Any, Dict, List, Tuple

from omegaconf import OmegaConf
import argparse
import torch
import hydra
import json
# from pytorch_lightning.loggers import Logger
from omegaconf import DictConfig
import look2hear.metrics
import look2hear.models
import look2hear.losses
from look2hear.metrics import MetricsTracker
from pyannote.audio import Pipeline

def GaGNet_wav(ests, refs, n_fft, hop_length, win_length):
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
                                n_fft=n_fft,
                                hop_length=hop_length,
                                win_length=win_length,
                                window=torch.hann_window(win_length).to(refs.device),
                                length=refs.shape[-1],
                                return_complex=False)  # (B,L)
    return batch_est_wav

def TaylorWav(ests, refs, n_fft, hop_length, win_length):
    # est
    ests = ests.permute(0,3,2,1)
    batch_spec_mag, batch_spec_phase = torch.norm(ests, dim=-1)**2.0,\
                                        torch.atan2(ests[..., -1], ests[...,0])
    ests = torch.stack((batch_spec_mag*torch.cos(batch_spec_phase),
                                    batch_spec_mag*torch.sin(batch_spec_phase)), dim=-1)
    # import pdb; pdb.set_trace()
    ests = torch.complex(real=ests[..., 0], imag=ests[..., 1])
    # import pdb; pdb.set_trace()
    batch_est_wav = torch.functional.istft(ests,
                                n_fft=n_fft,
                                hop_length=hop_length,
                                win_length=win_length,
                                window=torch.hann_window(win_length).to(refs.device),
                                length=refs.shape[-1],
                                return_complex=False)  # (B,L)
    return batch_est_wav

def test(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    print(os.path.join(cfg.exp.dir, cfg.exp.name, "best_model.pth"))
    cfg.model.pop("_target_", None)
    model = look2hear.models.FastFullSubnet.from_pretrain(os.path.join(cfg.exp.dir, cfg.exp.name, "best_model.pth"), **cfg.model).cuda()
    
    modelname = cfg.exp.name.split("-")[0]
    
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name, "results/"), exist_ok=True)
    metrics = MetricsTracker(save_file=os.path.join(cfg.exp.dir, cfg.exp.name, "results/")+"metrics.csv")

    mix = torchaudio.load("tests/noise/mix.wav")[0]
    spks = torchaudio.load("tests/noise/s1.wav")[0]
    
    mix = mix.squeeze(0).cuda()
    spks = spks.cuda()
    
    with torch.no_grad():
        json_dicts = {}
        with open("tests/noise/json_data.json", "r") as f:
            json_dicts = json.load(f)
        speaker_start_end = json_dicts["source1"]["start_end_points"]
            
        for start, end in speaker_start_end:
            mix_tmp = mix[start:end].view(1, -1)
            spks_tmp = spks[:, start:end]
            ehn_out = model(mix_tmp)
            if modelname in ["FastFullband", "Fullband", "FullSubNet", "InterSubNet"]:
                ehn_out = look2hear.losses.fullband_loss.inference(ehn_out, cfg.model.n_fft, cfg.model.hop_length, cfg.model.win_length, mix_tmp.shape[-1])
            if modelname in ["GaGNet", "G2Net", ]:
                ehn_out = GaGNet_wav(ehn_out, spks_tmp, cfg.model.n_fft, cfg.model.hop_length, cfg.model.win_length)
            if modelname in ["TaylorSENet"]:
                ehn_out = TaylorWav(ehn_out, spks, cfg.model.n_fft, cfg.model.hop_length, cfg.model.win_length)
            if ehn_out.dim() == 3:
                ehn_out = ehn_out.squeeze(0)
            metrics(
                mix=mix_tmp, 
                clean=spks_tmp, 
                estimate=ehn_out, 
                key="test", 
                spks_id="s1", 
                start_idx=start, 
                end_idx=end
            )
        print(metrics.update())
        metrics.final()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_dir",
        default="./config.yaml",
        help="Full path to save best validation model",
    )
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.conf_dir)
    
    test(cfg)
    
    