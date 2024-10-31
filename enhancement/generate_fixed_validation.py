import look2hear.datas
import torchaudio
from itertools import combinations
from tqdm import tqdm
import os
import argparse

def generate_fixed_validation(raw_dir, save_dir, noise_type="noise", n_src=1, num_samples=300, is_mono=False):
    assert n_src == 1, "n_src should be equal to 1"
    sample_idx = 0
    test_datasets = look2hear.datas.movingdatamodule.MovingTrainDataset(
        speech_dir=raw_dir,
        num_samples=num_samples,
        num_spks=n_src,
        noise_type=noise_type,
        is_mono=is_mono,
    )

    for idx in tqdm(range(len(test_datasets))):
        mix_wav, speaker_wav = test_datasets[idx]
        os.makedirs(os.path.join(save_dir, f"sample{sample_idx}"), exist_ok=True)
        if mix_wav.dim() == 1:
            mix_wav = mix_wav.unsqueeze(0)
        torchaudio.save(os.path.join(save_dir, f"sample{sample_idx}", "mix.wav"), mix_wav, 16000)
        torchaudio.save(os.path.join(save_dir, f"sample{sample_idx}", "clean.wav"), speaker_wav[0:1,...] if speaker_wav.dim() == 2 else speaker_wav[0], 16000)
        sample_idx += 1
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, help="Path to the raw data directory")
    parser.add_argument("--save_dir", type=str, help="Path to the save directory")
    parser.add_argument("--noise_type", type=str, default="noise", help="Type of noise to add")
    parser.add_argument("--n_src", type=int, default=1, help="Number of sources")
    parser.add_argument("--num_samples", type=int, default=300, help="Number of samples to generate")
    parser.add_argument("--is_mono", action="store_true", help="If True, the output will be mono")
    
    args = parser.parse_args()
    
    generate_fixed_validation(args.raw_dir, args.save_dir, args.noise_type, args.n_src, args.num_samples, args.is_mono)