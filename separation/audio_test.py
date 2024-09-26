import os
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


def test(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    root =  "./test"
    test_datasets = look2hear.datas.movingdatamodule.MovingTestPhaseDataset(
        "./test-remix",
        cfg.datas.sample_rate,
        cfg.datas.num_spks,
        cfg.datas.is_mono,
        cfg.datas.noise_type
    )
    if cfg.model.get("_target_") == "look2hear.models.dptnet.DPTNetModel":
        model = hydra.utils.instantiate(cfg.model)
        conf = torch.load(os.path.join(cfg.exp.dir, cfg.exp.name, "best_model.pth"), map_location="cpu")
        model.load_state_dict(conf["state_dict"])
        model.cuda()
    else:
        cfg.model.pop("_target_", None)
        model = look2hear.models.ConvTasNet.from_pretrain(os.path.join(cfg.exp.dir, cfg.exp.name, "best_model.pth"), **cfg.model).cuda()
    vad_model = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token="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", cache_dir="./huggingface_models")
    initial_params = {"onset": 0.3, "offset": 0.2,
                  "min_duration_on": 0.0, "min_duration_off": 0.0}
    vad_model.instantiate(initial_params)
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name, "results/"), exist_ok=True)
    metrics = MetricsTracker(save_file=os.path.join(cfg.exp.dir, cfg.exp.name, "results/")+"metrics.csv")
    length = len(test_datasets)
    for idx in range(len(test_datasets)):
        mix, spks, mix_path = test_datasets[idx]
        
        # Split path
        mix_path_split = mix_path.split("/")
        folder = mix_path_split[-2]
        filename = mix_path_split[-1]
        spks_id = mix_path_split[-3].split("-")[1:]
        
        mix = mix.squeeze(0).cuda()
        spks = spks.cuda()
        mix_path = mix_path
        with torch.no_grad():
            vad_results = vad_model(os.path.join(mix_path, "mix.wav"))
            vad_lists = vad_results.get_timeline().support()
            for index, start_end in enumerate(vad_lists):
                try:
                    start = int(start_end.start * 16000)
                    end = int(start_end.end * 16000)
                    mix_tmp = mix[start:end].cpu()
                    spks_tmp = spks[:, start:end]
                    
                    if end - start <= 320:
                        continue
                    try:
                        # 尝试在 GPU 上运行
                        model.cuda()
                        ests_out = model(mix_tmp.unsqueeze(0).to("cuda"))
                        ests_out = ests_out.to("cuda")
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            model.cpu()
                            ests_out = model(mix_tmp.unsqueeze(0)).to("cuda")
                        else:
                            raise e

                    if ests_out.dim() == 3:
                        ests_out = ests_out.squeeze(0)

                    metrics(
                        mix=mix_tmp, 
                        clean=spks_tmp, 
                        estimate=ests_out, 
                        key=os.path.join(mix_path_split[-3], folder, filename), 
                        spks_id=spks_id, 
                        start_idx=start, 
                        end_idx=end
                    )
                finally:
                    torch.cuda.empty_cache()
                
        
        if idx % 10 == 0:
            dicts = metrics.update()
            print("Processed {}/{} samples, Average SDR: {:.2f}, Average SISNR: {:.2f}".format(idx, length, dicts["sdr"], dicts["si-snr"]))
    
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
    
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name), exist_ok=True)
    # 保存配置到新的文件
    OmegaConf.save(cfg, os.path.join(cfg.exp.dir, cfg.exp.name, "config.yaml"))
    
    test(cfg)
    