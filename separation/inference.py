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
from speechbrain.pretrained import EncoderClassifier

import torch


def process_audio_segments(ests_out_lists, ests_out_emb_lists):
    """Process and concatenate audio segments into two complete audio tracks using PyTorch.

    Args:
        ests_out_lists (list): List of audio segments, each segment is [s1, s2].
        ests_out_emb_lists (list): List of embeddings, each segment is [e1, e2].

    Returns:
        final_s1 (torch.Tensor): Concatenated audio for s1.
        final_s2 (torch.Tensor): Concatenated audio for s2.
    """
    # Initialize lists to store the reordered s1 and s2 audio segments
    s1_list = []
    s2_list = []

    # Get the first segment's audio and embeddings as reference
    first_s1, first_s2 = ests_out_lists[0]
    first_e1, first_e2 = ests_out_emb_lists[0]

    # Append the first segment's audio to the lists
    s1_list.append(first_s1)
    s2_list.append(first_s2)

    # Process each subsequent segment
    for i in range(1, len(ests_out_lists)):
        s1_i, s2_i = ests_out_lists[i]
        e1_i, e2_i = ests_out_emb_lists[i]

        # Compute similarities between embeddings using cosine similarity
        # Ensure the similarities are scalar values

        # Adjust the dimension based on your embeddings' shape
        # Here we assume embeddings are 1D tensors (vectors)
        sim11 = torch.nn.functional.cosine_similarity(first_e1, e1_i, dim=0)
        sim22 = torch.nn.functional.cosine_similarity(first_e2, e2_i, dim=0)
        sim12 = torch.nn.functional.cosine_similarity(first_e1, e2_i, dim=0)
        sim21 = torch.nn.functional.cosine_similarity(first_e2, e1_i, dim=0)

        # If the similarities are tensors with more than one element, reduce them to scalars
        if sim11.numel() > 1:
            sim11 = sim11.mean()
            sim22 = sim22.mean()
            sim12 = sim12.mean()
            sim21 = sim21.mean()

        # Alternatively, if embeddings are single-element tensors, extract the scalar value
        if sim11.numel() == 1:
            sim11 = sim11.item()
            sim22 = sim22.item()
            sim12 = sim12.item()
            sim21 = sim21.item()

        # Decide whether to swap s1 and s2 based on similarity
        if (sim11 + sim22) >= (sim12 + sim21):
            # Keep the original order
            s1_list.append(s1_i)
            s2_list.append(s2_i)
        else:
            # Swap s1 and s2 to maximize similarity alignment
            s1_list.append(s2_i)
            s2_list.append(s1_i)

    # Concatenate all s1 and s2 segments to form the final audio tracks
    final_s1 = torch.cat(s1_list, dim=1)
    final_s2 = torch.cat(s2_list, dim=1)

    return final_s1, final_s2

def test(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.model.get("_target_") == "look2hear.models.dptnet.DPTNetModel":
        model = hydra.utils.instantiate(cfg.model)
        conf = torch.load(os.path.join(cfg.exp.dir, cfg.exp.name, "best_model.pth"), map_location="cpu")
        model.load_state_dict(conf["state_dict"])
        model.cuda()
    else:
        cfg.model.pop("_target_", None)
        model = look2hear.models.ConvTasNet.from_pretrain(os.path.join(cfg.exp.dir, cfg.exp.name, "best_model.pth"), **cfg.model).cuda()
    vad_model = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token="AUTH_TOKEN", cache_dir="./huggingface_models")
    
    initial_params = {"onset": 0.3, "offset": 0.2,
                  "min_duration_on": 0.0, "min_duration_off": 0.0}
    vad_model.instantiate(initial_params)
    
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="./huggingface_models/speechbrain/spkrec-ecapa-voxceleb",
        use_auth_token="AUTH_TOKEN")

    mix = torchaudio.load("tests/noise/mix.wav")[0]
    mix = mix.squeeze(0).cuda()
    
    ests_out_lists = []
    ests_out_emb_lists = []
    
    with torch.no_grad():
        vad_results = vad_model("tests/noise/mix.wav")
        vad_lists = vad_results.get_timeline().support()
        for index, start_end in enumerate(vad_lists):
            try:
                start = int(start_end.start * 16000)
                end = int(start_end.end * 16000)
                mix_tmp = mix[start:end].cpu()
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
                ests_out_lists.append([ests_out[:, 0], ests_out[:, 1]])
                ests_out_emb_lists.append([classifier.encode_batch(ests_out[:, 0]).view(-1), classifier.encode_batch(ests_out[:, 1]).view(-1)])
            finally:
                torch.cuda.empty_cache()

    final_s1, final_s2 = process_audio_segments(ests_out_lists, ests_out_emb_lists)
    torchaudio.save("tests/noise/s1_est.wav", final_s1, 16000)
    torchaudio.save("tests/noise/s2_est.wav", final_s2, 16000)
    
    
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
    
    