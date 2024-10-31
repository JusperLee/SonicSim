import look2hear.datas
import torchaudio
from itertools import combinations
from tqdm import tqdm
import os
import argparse

def generate_fixed_test(raw_dir, noise_type="noise", n_src=2, is_mono=False):
    assert n_src <= 3, "n_src should be less than or equal to 3"
    
    combs = list(combinations(range(3), n_src))
    
    sample_idx = 0
    for comb in combs:
        test_datasets = look2hear.datas.movingdatamodule.MovingTestEvalDataset(
            speech_dir=raw_dir,
            num_spks=comb,
            noise_type=noise_type,
            is_mono=is_mono,
        )
        folder_name = "-".join([str(c) for c in comb])
        for idx in tqdm(range(len(test_datasets))):
            mix_wav, speaker_wav, filepath = test_datasets[idx]
            os.makedirs(os.path.join(filepath.replace("/test/", f'/test-{noise_type}/{noise_type}-{folder_name}/')), exist_ok=True)
            if mix_wav.dim() == 1:
                mix_wav = mix_wav.unsqueeze(0)
            torchaudio.save(os.path.join(filepath.replace("/test/", f'/test-{noise_type}/{noise_type}-{folder_name}/'), "mix.wav"), mix_wav, 16000)
            for spk in range(n_src):
                torchaudio.save(os.path.join(filepath.replace("/test/", f'/test-{noise_type}/{noise_type}-{folder_name}/'), f"s{spk+1}.wav"), speaker_wav[spk:spk+1,...] if speaker_wav.dim() == 2 else speaker_wav[spk], 16000)
            sample_idx += 1
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, help="Path to the raw data directory")
    parser.add_argument("--noise_type", type=str, default="noise", help="Type of noise to add")
    parser.add_argument("--n_src", type=int, default=2, help="Number of sources")
    parser.add_argument("--is_mono", action="store_true", help="If True, the output will be mono")
    
    args = parser.parse_args()
    
    generate_fixed_test(args.raw_dir, args.noise_type, args.n_src, args.is_mono)