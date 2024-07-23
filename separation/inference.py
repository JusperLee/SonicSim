###
# Author: Kai Li
# Date: 2024-01-22 01:16:22
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2024-01-24 00:05:10
###
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
# from pytorch_lightning.loggers import Logger
from omegaconf import DictConfig
import look2hear.metrics
import look2hear.models
from look2hear.metrics import MetricsTracker
from pyannote.audio import Pipeline

def test(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.model.get("_target_") == "look2hear.models.dptnet.DPTNetModel":
        model = hydra.utils.instantiate(cfg.model)
        conf = torch.load(os.path.join(cfg.exp.dir, cfg.exp.name, "best_model.pth"), map_location="cpu")
        model.load_state_dict(conf["state_dict"])
        model.cuda()
    else:
        cfg.model.pop("_target_", None)
        model = look2hear.models.ConvTasNet.from_pretrain(os.path.join(cfg.exp.dir, cfg.exp.name, "best_model.pth"), **cfg.model).cuda()
    vad_model = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token="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", cache_dir="/home/likai/data5/huggingface_models")
    initial_params = {"onset": 0.3, "offset": 0.2,
                  "min_duration_on": 0.0, "min_duration_off": 0.0}
    vad_model.instantiate(initial_params)
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name, "results/"), exist_ok=True)
    metrics = MetricsTracker(save_file=os.path.join(cfg.exp.dir, cfg.exp.name, "results/")+"metrics.csv")

    mix = torchaudio.load("tests/noise/mix.wav")[0]
    spk = [torchaudio.load("tests/noise/s1.wav")[0], torchaudio.load("tests/noise/s2.wav")[0]]
    spks = torch.cat(spk, dim=0)
    
    mix = mix.squeeze(0).cuda()
    spks = spks.cuda()
    
    with torch.no_grad():
        vad_results = vad_model("tests/noise/mix.wav")
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
                    model.cuda()
                    ests_out = model(mix_tmp.unsqueeze(0).to("cuda"))
                    ests_out = ests_out.to("cuda")
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print("CUDA内存不足，切换到CPU。")
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
                    key="test", 
                    spks_id=["s1", "s2"], 
                    start_idx=start, 
                    end_idx=end
                )
            finally:
                torch.cuda.empty_cache()
                
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
    
    