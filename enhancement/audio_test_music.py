###
# Author: Kai Li
# Date: 2024-01-22 01:16:22
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2024-01-24 00:05:10
###
import os
os.environ['WANDB_API_KEY'] = "ca76d47c4da23aa9ceb8788307ab090ef5f5713c"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import warnings
warnings.filterwarnings("ignore")
import torchaudio

from rich import print
import json
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf
import argparse
import pandas as pd
import pytorch_lightning as pl
import scipy as sp
from sympy import im
import torch
import hydra
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
# from pytorch_lightning.loggers import Logger
from omegaconf import DictConfig
import look2hear.system
import look2hear.datas
import look2hear.metrics
import look2hear.models
import look2hear.losses
from look2hear.metrics import MetricsTracker
from pyannote.audio import Inference, Model, Pipeline
from look2hear.utils import RankedLogger, instantiate, print_only
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.model import Model, Specifications
from pyannote.audio.core.task import Resolution
from pyannote.audio.utils.multi_task import map_with_specifications
from pyannote.audio.utils.permutation import mae_cost_func, permutate
from pyannote.audio.utils.powerset import Powerset
from pyannote.audio.utils.reproducibility import fix_reproducibility
from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature
from typing import Callable, List, Optional, Text, Tuple, Union
import numpy as np

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
    root =  "/home/likai/data5/moving-data/test"
    test_datasets = look2hear.datas.movingdatamodule_remix.MovingTestPhaseDataset(
        "/home/likai/data5/moving-data/enh-test-remix-music",
        cfg.datas.sample_rate,
        cfg.datas.num_spks,
        cfg.datas.is_mono,
        "music"
    )
    if cfg.model.get("_target_") == "look2hear.models.dptnet.DPTNetModel":
        model = hydra.utils.instantiate(cfg.model)
        conf = torch.load(os.path.join(cfg.exp.dir, cfg.exp.name, "best_model.pth"), map_location="cpu")
        model.load_state_dict(conf["state_dict"])
        model.cuda()
    else:
        cfg.model.pop("_target_", None)
        model = look2hear.models.Fullband.from_pretrain(os.path.join(cfg.exp.dir, cfg.exp.name, "best_model.pth"), **cfg.model).cuda()
    
    # start_end_dicts = {}
    # with open("tests/ehn_noise_start_end_all.json", "r") as f:
    #     start_end_dicts = json.load(f)
    
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name, "results/"), exist_ok=True)
    metrics = MetricsTracker(save_file=os.path.join(cfg.exp.dir, cfg.exp.name, "results/")+"metrics.csv")
    length = len(test_datasets)
    for idx in range(len(test_datasets)):
        mix, spks, mix_path = test_datasets[idx]
        
        # Split path
        mix_path_split = mix_path.split("/")
        folder = mix_path_split[-2]
        filename = mix_path_split[-1]
        spks_id = mix_path_split[-3]
        # import pdb; pdb.set_trace()
        mix = mix.squeeze(0).cuda()
        spks = spks.cuda()
        mix_path = mix_path
        with torch.no_grad():
            # from wav_inference_fullband import wav_chunk_inference
            # ests_out = wav_chunk_inference(model, mix, target_length=4, hop_length=1.5, batch_size=1, n_tracks=1)
            json_dicts = {}
            with open(f"/home/likai/data5/moving-data/test/{folder}/{filename}/json_data.json", "r") as f:
                json_dicts = json.load(f)
            speaker_start_end = json_dicts[f"source{spks_id}"]["start_end_points"]
            
            for start, end in speaker_start_end:
                # import pdb; pdb.set_trace()
                mix_tmp = mix[start:end].view(1, -1)
                spks_tmp = spks[:, start:end]
                ehn_out = model(mix_tmp)
                # ehn_out = GaGNet_wav(ehn_out, spks_tmp, cfg.loss.n_fft, cfg.loss.hop_length, cfg.loss.win_length)
                # ehn_out = look2hear.losses.fullband_loss.inference(ehn_out, cfg.loss.n_fft, cfg.loss.hop_length, cfg.loss.win_length, mix_tmp.shape[-1])
                # os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name, "results/", mix_path_split[-3], folder, filename), exist_ok=True)
                # torchaudio.save(os.path.join(cfg.exp.dir, cfg.exp.name, "results/", mix_path_split[-3], folder, filename, f"{start}_{end}.wav"), ehn_out.cpu().view(1, -1), cfg.datas.sample_rate)
                # import pdb; pdb.set_trace()
                if ehn_out.dim() == 3:
                    ehn_out = ehn_out.squeeze(0)

                metrics(
                    mix=mix_tmp, 
                    clean=spks_tmp, 
                    estimate=ehn_out, 
                    key=os.path.join(mix_path_split[-3], folder, filename), 
                    spks_id=spks_id, 
                    start_idx=start, 
                    end_idx=end
                )
                # metrics.final()
                # import pdb; pdb.set_trace()
        if idx % 10 == 0:
            dicts = metrics.update()
            print(dicts)
    
    metrics.final()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_dir",
        default="/home/likai/data5/Exps/convtasnet-moxing-data-lk-valrmix/config.yaml",
        help="Full path to save best validation model",
    )
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.conf_dir)
    
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name), exist_ok=True)
    # 保存配置到新的文件
    OmegaConf.save(cfg, os.path.join(cfg.exp.dir, cfg.exp.name, "config.yaml"))
    
    test(cfg)
    