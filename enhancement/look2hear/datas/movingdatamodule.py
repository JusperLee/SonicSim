import os
import json
import librosa
import numpy as np
from typing import Any, Tuple
import scipy
import soundfile as sf
import torch
import random
from collections import defaultdict
from pytorch_lightning import LightningDataModule
import torchaudio
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from typing import Any, Dict, Optional, Tuple
from pytorch_lightning.utilities import rank_zero_only
EPS = np.finfo(float).eps

@rank_zero_only
def print_(message: str):
    print(message)
    
def find_bottom_directories(root_dir):
    bottom_directories = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if not dirnames:
            bottom_directories.append(dirpath)
    return bottom_directories

def compute_mch_rms_dB(mch_wav, fs=16000, energy_thresh=-50):
    """Return the wav RMS calculated only in the active portions"""
    mean_square = max(1e-20, torch.mean(mch_wav ** 2))
    return 10 * np.log10(mean_square)

def overlap_audio(waveform, sample_rate, delay=6):
    delay_samples = int(delay * sample_rate)
    
    # Create padded versions of the waveform
    padded_waveform_forward = torch.nn.functional.pad(waveform, (delay_samples, 0))
    padded_waveform_backward = torch.nn.functional.pad(waveform, (0, delay_samples))
    
    # Truncate the padded waveforms to the original length
    padded_waveform_forward = padded_waveform_forward[:, :waveform.size(1)]
    padded_waveform_backward = padded_waveform_backward[:, -waveform.size(1):]
    
    # Combine the waveforms
    overlapped_waveform = padded_waveform_forward + padded_waveform_backward + waveform
    
    return overlapped_waveform

def find_overlap_region(data, min_overlap=2, max_overlap=3, max_duration=None, sample_rate=None):
    all_points = []
    for source in data.values():
        if 'start_end_points' in source:
            all_points.extend(source['start_end_points'])


    min_start = min(point[0] for point in all_points)
    max_end = max(point[1] for point in all_points)

    while True:
        overlap_start = random.randint(min_start, max_end)
        overlap_end = random.randint(overlap_start, max_end)

        if max_duration is not None and sample_rate is not None:
            duration = (overlap_end - overlap_start) / sample_rate
            if duration < max_duration:
                continue

        overlap_count = sum(
            overlap_start <= point[0] <= overlap_end or overlap_start <= point[1] <= overlap_end
            for point in all_points
        )

        if min_overlap <= overlap_count <= max_overlap:
            return overlap_start, overlap_end
    
class MovingTrainDataset(Dataset):
    def __init__(
        self, 
        speech_dir: str,
        sample_rate: int = 16000,
        duration: float = 4.0,
        num_samples: int = 1000,
        num_spks: int = 2,
        is_mono: bool = True,
        noise_type: str = "noise",
    ) -> None:
        self.data_dirs = find_bottom_directories(speech_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = num_samples
        self.num_spks = num_spks
        self.is_mono = is_mono
        self.noise_type = noise_type
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        speech_dir = random.choice(self.data_dirs)
        speaker_wavs = []
        speaker_id = random.sample(range(1, 4), self.num_spks)
        for idx in speaker_id:
            speaker_wav, _ = torchaudio.load(speech_dir + '/moving_audio_{}.wav'.format(idx))
            if self.is_mono:
                speaker_wav = speaker_wav.mean(dim=0)
            speaker_wavs.append(speaker_wav)
        speaker_wav = torch.stack(speaker_wavs)
        
        noise_wavs = []
        
        if self.noise_type == "all":
            noise_types = ["music", "noise"]
        else:
            noise_types = [self.noise_type]
        
        for noise in noise_types:
            noise_wav, _ = torchaudio.load(speech_dir + '/{}_audio.wav'.format(noise))
            if self.is_mono:
                noise_wav = noise_wav.mean(dim=0)
            noise_wavs.append(noise_wav)
        noise_wav = torch.stack(noise_wavs)
        
        start = 0
        end = 0
        for_idx = 0
        while True:
            if for_idx > 100:
                break
            start = random.randint(0, speaker_wav.shape[-1] - self.sample_rate * self.duration)
            end = int(start + self.sample_rate * self.duration)

            speaker_wav_tmp = speaker_wav[..., start:end]
            # import pdb; pdb.set_trace()
            is_silence = 0
            for i in range(self.num_spks):
                if compute_mch_rms_dB(speaker_wav_tmp[i]) < -40:
                    is_silence = 1
            if is_silence == 1:
                for_idx += 1
                continue
            break
        
        # print("start: ", start, "end: ", end)
        speaker_wav = speaker_wav[..., start:end]
        noise_wav = noise_wav[..., start:end]
        
        # Random SIR and SNR
        sirs = torch.Tensor(self.num_spks-1).uniform_(-6,6).numpy()
        target_refch_energy = compute_mch_rms_dB(speaker_wav[0])

        for i in range(self.num_spks-1):
            sir = sirs[i]
            intf_refch_energy = compute_mch_rms_dB(speaker_wav[i+1])
            gain = min(target_refch_energy - intf_refch_energy - sir, 40)
            speaker_wav[i + 1] *= 10. ** (gain / 20.)
        
        all_speech = torch.sum(speaker_wav, dim=0)
        all_noise = torch.sum(noise_wav, dim=0)
        
        target_refch_energy = compute_mch_rms_dB(all_speech)
        snr = torch.Tensor(1).uniform_(10, 20).numpy()
        noise_refch_energy = compute_mch_rms_dB(all_noise)
        gain = min(target_refch_energy - noise_refch_energy - snr, 40)
        all_noise *= 10.**(gain/20.)
        
        mix_wav = all_speech + all_noise
        
        return mix_wav, speaker_wav.squeeze(0)
    
class MovingEvalDataset(Dataset):
    def __init__(
        self, 
        speech_dir: str,
        sample_rate: int = 16000,
        num_spks: int = 2,
        is_mono: bool = True,
        noise_type: str = "noise",
    ) -> None:
        self.speech_dir = speech_dir
        self.data = os.listdir(speech_dir)
        self.sample_rate = sample_rate
        self.num_spks = num_spks
        self.is_mono = is_mono
        self.noise_type = noise_type
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        folder = self.data[idx]
        speaker_wav, _ = torchaudio.load(os.path.join(self.speech_dir, folder, 'clean.wav'))
        mix_wav, _ = torchaudio.load(os.path.join(self.speech_dir, folder, 'mix.wav'))
        if self.is_mono:
            mix_wav = mix_wav.mean(dim=0)
        return mix_wav, speaker_wav
    
class MovingTestEvalDataset(Dataset):
    def __init__(
        self, 
        speech_dir: str,
        sample_rate: int = 16000,
        num_spks: int = 0,
        is_mono: bool = True,
        noise_type: str = "noise",
    ) -> None:
        self.data_dirs = find_bottom_directories(speech_dir)
        # self.data = os.listdir(speech_dir)
        self.sample_rate = sample_rate
        self.num_spks = num_spks
        self.is_mono = is_mono
        self.noise_type = noise_type
        
    def __len__(self) -> int:
        return len(self.data_dirs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        folder = self.data_dirs[idx]
        speaker_wavs, _ = torchaudio.load(os.path.join(folder, 'moving_audio_{}.wav'.format(self.num_spks+1)))
        if self.is_mono:
            speaker_wavs = speaker_wavs.mean(dim=0)
        
        noise_wavs = []
        if self.noise_type == "all":
            noise_types = ["music", "noise"]
        else:
            noise_types = [self.noise_type]
        
        for noise in noise_types:
            noise_wav, _ = torchaudio.load(os.path.join(folder, '{}_audio.wav'.format(noise)))
            if self.is_mono:
                noise_wav = noise_wav.mean(dim=0)
            noise_wavs.append(noise_wav)
        noise_wav = torch.stack(noise_wavs)
        all_noise = torch.sum(noise_wav, dim=0)
        # # Random SIR and SNR
        target_refch_energy = compute_mch_rms_dB(speaker_wavs)
        
        all_noise = torch.sum(noise_wav, dim=0)
        all_noise = overlap_audio(all_noise.view(1, -1), self.sample_rate, delay=6).view(-1)
        
        if "music" in noise_types:
            target_refch_energy = compute_mch_rms_dB(speaker_wavs)
            snr = torch.Tensor(1).uniform_(-10, 15).numpy()
            noise_refch_energy = compute_mch_rms_dB(all_noise)
            gain = min(target_refch_energy - noise_refch_energy - snr, 40)
            all_noise *= 10.**(gain/20.)
        else:
            target_refch_energy = compute_mch_rms_dB(speaker_wavs)
            snr = torch.Tensor(1).uniform_(-10, 15).numpy()
            # print(snr)
            noise_refch_energy = compute_mch_rms_dB(all_noise)
            gain = min(target_refch_energy - noise_refch_energy - snr, 40)
            all_noise *= 10.**(gain/20.)
        
        # # 生成混合语音
        mix_wav = speaker_wavs + all_noise
        
        
        return mix_wav, speaker_wavs, os.path.join(folder)

class MovingTestDataset(Dataset):
    def __init__(
        self, 
        speech_dir: str,
        sample_rate: int = 16000,
        num_spks: int = 2,
        is_mono: bool = True,
        noise_type: str = "noise",
    ) -> None:
        self.data_dirs = find_bottom_directories(speech_dir)
        # self.data = os.listdir(speech_dir)
        self.sample_rate = sample_rate
        self.num_spks = num_spks
        self.is_mono = is_mono
        self.noise_type = noise_type
        
    def __len__(self) -> int:
        return len(self.data_dirs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        folder = self.data_dirs[idx]
        speaker_wav, _ = torchaudio.load(os.path.join(folder, 'clean.wav'))
        mix_wav, _ = torchaudio.load(os.path.join(folder, 'mix.wav'))
        if self.is_mono:
            mix_wav = mix_wav.mean(dim=0)
        return mix_wav.view(-1), speaker_wav.view(-1)
    
class MovingTestPhaseDataset(Dataset):
    def __init__(
        self, 
        speech_dir: str,
        sample_rate: int = 16000,
        num_spks: int = 2,
        is_mono: bool = True,
        noise_type: str = "noise",
    ) -> None:
        self.data_dirs = find_bottom_directories(speech_dir)
        # self.data = os.listdir(speech_dir)
        self.sample_rate = sample_rate
        self.num_spks = num_spks
        self.is_mono = is_mono
        self.noise_type = noise_type
        
    def __len__(self) -> int:
        return len(self.data_dirs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        folder = self.data_dirs[idx]
        speaker_wavs = []
        for i in range(self.num_spks):
            speaker_wav, _ = torchaudio.load(os.path.join(folder, 's{}.wav'.format(i+1)))
            if self.is_mono:
                speaker_wav = speaker_wav.mean(dim=0)
            speaker_wavs.append(speaker_wav)
        speaker_wav = torch.stack(speaker_wavs)
        
        mix = torchaudio.load(os.path.join(folder, 'mix.wav'))[0]
        
        return mix, speaker_wav, os.path.join(folder)

class MovingDataModuleRemix(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        test_dir: str,
        sample_rate: int = 16000,
        duration: float = 4.0,
        num_samples: int = 1000,
        num_spks: int = 2,
        batch_size: int = 32,
        num_workers: int = 4,
        is_mono: bool = True,
        noise_type: str = "noise",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = MovingTrainDataset(
                speech_dir=self.hparams.train_dir,
                sample_rate=self.hparams.sample_rate,
                duration=self.hparams.duration,
                num_samples=self.hparams.num_samples,
                num_spks=self.hparams.num_spks,
                is_mono=self.hparams.is_mono,
                noise_type=self.hparams.noise_type
            )
            self.data_val = MovingTestDataset(
                speech_dir=self.hparams.val_dir,
                sample_rate=self.hparams.sample_rate,
                num_spks=self.hparams.num_spks,
                is_mono=self.hparams.is_mono,
                noise_type=self.hparams.noise_type
            )
            self.data_test = MovingTestDataset(
                speech_dir=self.hparams.test_dir,
                sample_rate=self.hparams.sample_rate,
                num_spks=self.hparams.num_spks,
                is_mono=self.hparams.is_mono,
                noise_type=self.hparams.noise_type
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
        )
        
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
        )
        