import os
import json
import torch
import torchaudio
import matplotlib.pyplot as plt
import torch.nn.functional as F
import typing as T
import itertools
import random
import pyloudnorm as pyln
import numpy as np
import math
from rich import print

from ss_utils import render_rir_parallel

def fft_conv(
    signal: torch.Tensor,
    kernel: torch.Tensor,
    is_cpu: bool = False
) -> torch.Tensor:
    """
    Perform convolution of a signal and a kernel using Fast Fourier Transform (FFT).

    Args:
    - signal (torch.Tensor): Input signal tensor.
    - kernel (torch.Tensor): Kernel tensor.
    - is_cpu (bool, optional): Flag to determine if the operation should be on the CPU.

    Returns:
    - torch.Tensor: Convolved signal.
    """

    if is_cpu:
        signal = signal.detach().cpu()
        kernel = kernel.detach().cpu()

    padded_signal = F.pad(signal.reshape(-1), (0, kernel.size(-1) - 1))
    padded_kernel = F.pad(kernel.reshape(-1), (0, signal.size(-1) - 1))

    signal_fr = torch.fft.rfftn(padded_signal, dim=-1)
    kernel_fr = torch.fft.rfftn(padded_kernel, dim=-1)

    output_fr = signal_fr * kernel_fr
    output = torch.fft.irfftn(output_fr, dim=-1)

    return output

def normalize(audio, norm='peak'):
    if norm == 'peak':
        peak = abs(audio).max()
        if peak != 0:
            return audio / peak
        else:
            return audio
    elif norm == 'rms':
        if torch.is_tensor(audio):
            audio = audio.numpy()
        audio_without_padding = np.trim_zeros(audio, trim='b')
        rms = np.sqrt(np.mean(np.square(audio_without_padding))) * 100
        if rms != 0:
            return audio / rms
        else:
            return audio
    else:
        raise NotImplementedError

def lufs_norm(data, sr, norm=-6):
    block_size = 0.4 if len(data) / sr >= 0.4 else len(data) / sr
    # measure the loudness first 
    meter = pyln.Meter(rate=sr, block_size=block_size)
    loudness = meter.integrated_loudness(data)
    # import pdb; pdb.set_trace()
    if math.isinf(loudness):
        loudness = -40
        print("loudness is inf")

    norm_data = pyln.normalize.loudness(data, loudness, norm)
    n, d = np.sum(np.array(norm_data)), np.sum(np.array(data))
    gain = n/d if d else 0.0

    return norm_data, gain

def get_lufs_norm_audio(audio, sr=16000, lufs=-6):
    class_lufs = np.random.uniform(lufs-2, lufs+2)
    data_norm, gain = lufs_norm(data=audio,sr=sr,norm=class_lufs)
    return data_norm, gain

def all_pairs(
    list1: T.List[T.Any],
    list2: T.List[T.Any]
) -> T.Tuple[T.List[T.Any], T.List[T.Any]]:
    """
    Computes all pairs of combinations between two lists.

    Args:
    - list1: First list.
    - list2: Second list.

    Returns:
    - Two lists containing paired elements from list1 and list2.
    """

    list_pair = list(itertools.product(list1, list2))

    list1_pair, list2_pair = zip(*list_pair)
    list1_pair = list(list1_pair)
    list2_pair = list(list2_pair)

    return list1_pair, list2_pair

def clip_all(audio_list):
    """
    Clips all audio signals in a list to the same length.

    Args: 
        audio_list: List of audio signals.

    Returns: 
    - List of audio signals of the same length.
    """

    min_length = min(audio.shape[-1] for audio in audio_list)
    clipped_audio_list = []
    for audio in audio_list:
        clipped_audio = audio[..., :min_length]
        clipped_audio_list.append(clipped_audio)

    return clipped_audio_list

def clip_two(audio1, audio2):
    """
    Clips two audio signals to the same length.

    Args:
        audio1: First audio signal.
        audio2: Second audio signal.

    Returns: 
    - Two audio signals of the same length.
    """

    length_diff = audio1.shape[-1] - audio2.shape[-1]

    if length_diff == 0:
        return audio1, audio2
    elif length_diff > 0:
        audio1 = audio1[..., :audio2.shape[-1]]
    elif length_diff < 0:
        audio2 = audio2[..., :audio1.shape[-1]]

    return audio1, audio2

def get_random_wav_path(
    audio_dir: str,
    length: int,
    threshold: float = 0.9,
) -> T.List[str]:
    """
    Gets a random selection of audio files from a directory whose total duration is between 90% and 100% of a given length.

    Args:
    - audio_dir: Path to the directory containing audio files.
    - length: Desired total length of the audio selection.
    - threshold: Threshold for the total length of the audio selection.
    - audio_format: Format of the audio files.

    Returns:
    - List of paths to randomly selected audio files.
    """

    # 获取目录中所有音频文件的路径
    audio_path_list = []
    # import pdb; pdb.set_trace()
    for root, _, files in os.walk(str(audio_dir)):
        for file in files:
            if not file.endswith('.txt'):
                audio_path_list.append(os.path.join(root, file))
    print(f"audio_path_list: {len(audio_path_list)}") 
    # 计算每个音频文件的长度
    audio_lengths = {path: torchaudio.load(path)[0].shape[-1] for path in audio_path_list}

    # 随机选择音频，直到总长度在给定范围内
    selected_paths = []
    current_length = 0
    min_length = length * threshold
    max_length = length
    # import pdb; pdb.set_trace()
    while audio_path_list and current_length < min_length:
        path = random.choice(audio_path_list)
        if current_length + audio_lengths[path] <= max_length:
            selected_paths.append(path)
            current_length += audio_lengths[path]
        else:
            # selected_paths.append(path)
            break
        audio_path_list.remove(path)

    return selected_paths

def get_random_wav_path_from_json(
    json_dir: str,
    length: int,
    threshold: float = 0.9,
) -> T.List[str]:
    """
    Gets a random selection of audio files from a directory whose total duration is between 90% and 100% of a given length.

    Args:
    - json_dir: Path to the directory containing audio files.
    - length: Desired total length of the audio selection.
    - threshold: Threshold for the total length of the audio selection.
    - audio_format: Format of the audio files.

    Returns:
    - List of paths to randomly selected audio files.
    """
    # 计算每个音频文件的长度
    with open(json_dir) as f:
        audio_lengths = json.load(f)
    audio_path_list = list(audio_lengths.keys())
    # import pdb; pdb.set_trace()
    # 随机选择音频，直到总长度在给定范围内
    selected_paths = []
    current_length = 0
    min_length = length * threshold
    max_length = length

    while audio_path_list and current_length < min_length:
        path = random.choice(audio_path_list)
        if current_length + audio_lengths[path] < max_length:
            selected_paths.append(path)
            current_length += audio_lengths[path]
        else:
            selected_paths.append(path)
            break
        audio_path_list.remove(path)

    return selected_paths

def create_long_audio(audio_path: str, length: float, sample_rate: int = 16000) -> torch.Tensor:
    """
    Concatenates audio signals in a list to create a long audio signal.

    Args:
        audio_path: List of audio signals.
        length: Length of the long audio signal.

    Returns: Long audio signal.
    """
    print("create_long_audio: ", audio_path)
    # import pdb; pdb.set_trace()
    audio_path_list = get_random_wav_path(audio_path, int(length * sample_rate))
    audio_name = [os.path.basename(audio_path_) for audio_path_ in audio_path_list]
    audios = [torchaudio.load(audio_path_)[0] for audio_path_ in audio_path_list]
    long_audio = torch.zeros((1, int(length * sample_rate)), device=audios[0].device)
    
    start_end_points = []
    audio_name_list = []
    current_duration = 0
    while current_duration < int(length * sample_rate):
        if len(audios) == 0:
            break
        random_idx = random.randint(0, len(audios)-1)
        audio = audios[random_idx]
        
        silence = torch.zeros((1, random.randint(0, int(10 * sample_rate))), device=audio.device)
        audio = torch.cat([silence, audio], dim=-1)
        
        # if audio.shape[-1] >= int(length * sample_rate) - current_duration:
        #     audios.pop(random_idx)
        #     continue
        
        if current_duration + audio.shape[-1] <= int(length * sample_rate):
            start_end_points.append((current_duration+silence.shape[-1], current_duration+audio.shape[-1]))
            long_audio[:, current_duration:current_duration+audio.shape[-1]] += audio
            current_duration += audio.shape[-1]
            audio_name_list.append(audio_name[random_idx])
            audios.pop(random_idx)
            audio_name.pop(random_idx)
        else:
            break
        
    return long_audio, start_end_points, audio_name_list

def create_background_audio(audio_path: str, length: float, sample_rate: int = 16000) -> torch.Tensor:
    """
    Concatenates audio signals in a list to create a long audio signal.
    
    Args:
        audio_path: List of audio signals.
        length: Length of the long audio signal.
        sample_rate: Sample rate of the audio signal.
        
    Returns: Long audio signal.
    """
    print("create_background_audio: ", audio_path)
    audio_path_list = get_random_wav_path_from_json(audio_path, int(length * sample_rate), threshold=0.4)
    audio_name = [os.path.basename(audio_path) for audio_path in audio_path_list]
    audios = [torchaudio.load(audio_path)[0] for audio_path in audio_path_list]
    long_audio = torch.zeros((1, int(length * sample_rate)), device=audios[0].device)
    
    start_end_points = []
    audio_name_list = []
    current_duration = 0
    while current_duration < int(length * sample_rate):
        if len(audios) == 0:
            break
        random_idx = random.randint(0, len(audios)-1)
        audio = audios[random_idx]
        
        if audio.shape[0] == 2:
            audio = audio.mean(dim=0, keepdim=True)
        
        silence = torch.zeros((1, random.randint(0, int(10 * sample_rate))), device=audio.device)
        audio = torch.cat([audio, silence], dim=-1)
        
        if audio.shape[-1] >= int(length * sample_rate) - current_duration:
            random_start = random.randint(0, (int((length * sample_rate - current_duration)*0.1)))
            random_end = random.randint(0, (int((length * sample_rate - current_duration)*0.1)))
            start_end_points.append((random_start+current_duration, int(length * sample_rate) - random_end))
            try:
                long_audio[:, random_start+current_duration:int(length * sample_rate) - random_end] += audio[:, random_start:int(length * sample_rate) - random_end - current_duration]
            except:
                import pdb; pdb.set_trace()
            break
        
        if current_duration + audio.shape[-1] < int(length * sample_rate):
            start_end_points.append((current_duration, current_duration+audio.shape[-1]))
            long_audio[:, current_duration:current_duration+audio.shape[-1]] += audio
            current_duration += audio.shape[-1]
            audio_name_list.append(audio_name[random_idx])
            audios.pop(random_idx)
        else:
            break
            
    return long_audio, start_end_points, audio_name_list

def generate_rir_combination(
    room: str,
    source_idx_list: T.List[T.Any],
    receiver_idx_list: T.List[T.Any],
    receiver_rotation_list: T.List[float],
    channel_type: str = 'Binaural',
    channel_order: int = 0
) -> T.List[T.List[torch.Tensor]]:
    """
    Generates room impulse responses (RIR) for given source and receiver combinations.

    Args:
    - room:                     Room object for which RIRs need to be computed.
    - source_idx_list:          List of source indices.
    - grid_points_source:       Grid points for the source.
    - receiver_idx_list:        List of receiver indices.
    - receiver_rotation_list:   List of receiver rotations.
    - grid_points_receiver:     Grid points for the receiver.
    - channel_type:             Type of the channel. Defaults to 'Ambisonics'.
    - channel_order:            Order of the channel for Ambisonics. Defulats to 0, as video usually does not support HOA.

    Returns:
    - A 2D list containing RIRs for every source-receiver combination.
    """

    # Set source and receiver points
    source_point_list = source_idx_list
    receiver_point_list = receiver_idx_list

    source_points_pair, receiver_points_pair = all_pairs(source_point_list, receiver_point_list)
    # import pdb; pdb.set_trace()
    _, receiver_rotation_pair = all_pairs(source_point_list, receiver_rotation_list)

    room_list = [room] * len(source_points_pair)
    filename_list = None

    # Render RIR for grid points
    ir_list = render_rir_parallel(
        room_list, 
        source_points_pair, 
        receiver_points_pair, 
        receiver_rotation_list=receiver_rotation_pair,
        filename_list=filename_list, 
        channel_type=channel_type, 
        channel_order=channel_order
    )
    
    # import pdb; pdb.set_trace()
    ir_list = clip_all(ir_list)  # make the length consistent
    num_channel = len(ir_list[0])

    # Reshape RIR
    num_sources = len(source_idx_list)
    num_receivers = len(receiver_idx_list)
    ir_output = torch.stack(ir_list).reshape(num_sources, num_receivers, num_channel, -1)  # '-1' will infer the remaining dimension based on the size of each tensor in ir_list
    ir_output /= ir_output.abs().max()

    return ir_output