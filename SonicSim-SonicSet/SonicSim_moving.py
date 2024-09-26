#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import torch
import numpy as np
import typing as T
from scipy import signal
from scipy.signal import oaconvolve

from SonicSim_rir import Receiver, Source, Scene


def setup_dynamic_interp(
    receiver_position: np.ndarray,
    total_samples: int,
) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Setup moving path with a constant speed for a receiver, given its positions in 3D space.

    Args:
    - receiver_position:    Receiver positions in 3D space of shape (num_positions, 3).
    - total_samples:        Total number of samples in the audio.

    Returns:
    - interp_index:         Indices representing the start positions for interpolation.
    - interp_weight:        Weight values for linear interpolation.
    """

    # Calculate the number of samples per interval
    distance = np.linalg.norm(np.diff(receiver_position, axis=0), axis=1)
    speed_per_sample = distance.sum() / total_samples
    samples_per_interval = np.round(distance / speed_per_sample).astype(int)

    # Distribute rounding errors
    error = total_samples - samples_per_interval.sum()
    for i in np.random.choice(len(samples_per_interval), abs(error)):
        samples_per_interval[i] += np.sign(error)

    # Calculate indices and weights for linear interpolation
    interp_index = np.repeat(np.arange(len(distance)), samples_per_interval)
    interp_weight = np.concatenate([np.linspace(0, 1, num, endpoint=False) for num in samples_per_interval])

    return interp_index, interp_weight.astype(np.float32)

def convolve_fixed_receiver(
    source_audio: np.ndarray,
    rirs: np.ndarray) -> np.ndarray:
    """
    Apply convolution between an audio signal and fixed impulse responses (IRs).
    
    Args:
    - source_audio:     Source audio of shape (audio_len,)
    - rirs:             RIRs of shape (num_channels, ir_length)
    
    Returns:
    - Convolved audio signal of shape (num_channels, audio_len)
    """
    reverb_wav = signal.fftconvolve(source_audio.reshape(1, -1), rirs, mode="full")[:, : source_audio.shape[-1]]
    return reverb_wav

def convolve_moving_receiver(
    source_audio: np.ndarray,
    rirs: np.ndarray,
    interp_index: T.List[int],
    interp_weight: T.List[float]
) -> np.ndarray:
    """
    Apply convolution between an audio signal and moving impulse responses (IRs).

    Args:
    - source_audio:     Source audio of shape (audio_len,)
    - rirs:             RIRs of shape (num_positions, num_channels, ir_length)
    - interp_index:     Indices representing the start positions for interpolation of shape (audio_len,).
    - interp_weight:    Weight values for linear interpolation of shape (audio_len,).

    Returns:
    - Convolved audio signal of shape (num_channels, audio_len)
    """

    num_channels = rirs.shape[1]
    audio_len = source_audio.shape[0]

    # Perform convolution for each position and channel
    convolved_audios = oaconvolve(source_audio[None, None, :], rirs, axes=-1)[..., :audio_len]

    # NumPy fancy indexing and broadcasting for interpolation
    start_audio = convolved_audios[interp_index, np.arange(num_channels)[:, None], np.arange(audio_len)]
    end_audio = convolved_audios[interp_index + 1, np.arange(num_channels)[:, None], np.arange(audio_len)]
    interp_weight = interp_weight[None, :]

    # Apply linear interpolation
    moving_audio = (1 - interp_weight) * start_audio + interp_weight * end_audio

    return moving_audio

def interpolate_moving_audio(
    source1_audio: torch.Tensor,
    ir1_list: T.List[torch.Tensor],
    receiver_position: np.ndarray
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Interpolates audio for a moving receiver.

    Args:
    - source1_audio:        First source audio.
    - source2_audio:        Second source audio.
    - ir1_list:             List of impulse responses for source 1.
    - ir2_list:             List of impulse responses for source 2.
    - receiver_position:    Positions of the moving receiver.

    Returns:
    - Tuple containing combined audio, interpolated audio from source 1, and interpolated audio from source 2.
    """

    # Prepare for interpolation
    audio_len = source1_audio.shape[-1]
    interp_index, interp_weight = setup_dynamic_interp(np.array(receiver_position), audio_len)
    # import pdb; pdb.set_trace()
    # Generate audio for moving receiver
    receiver_audio_1 = convolve_moving_receiver(source1_audio.numpy()[0], np.array(ir1_list).squeeze(1), interp_index, interp_weight)
    receiver_audio_1 = receiver_audio_1[..., :source1_audio.shape[-1]]

    return torch.from_numpy(receiver_audio_1)

def interpolate_values(
    start: float,
    end: float,
    interp_weight: float
) -> float:
    """
    Interpolate between two values based on the weight values.

    Args:
    - start:            Beginning value.
    - end:              Ending value.
    - interp_weight:    Weight for linear interpolation

    Returns:
    - Interpolated value.
    """

    return (1 - interp_weight) * start + interp_weight * end

def interpolate_rgb_images(
    scene: Scene,
    receiver_position: torch.Tensor,
    receiver_rotation_list: T.List[float],
    video_len: int
) -> T.List[np.ndarray]:
    """
    Interpolates RGB images based on receiver movement and rotation.

    Args:
    - scene:                  Scene object to render the images from.
    - receiver_position:      Positions of the receiver along the path.
    - receiver_rotation_list: List of rotations for the receiver.
    - video_len:              Number of frames in the video.

    Returns:
    - List of interpolated RGB images.
    """

    interp_index, interp_weight = setup_dynamic_interp(receiver_position.numpy(), video_len)

    interpolated_rgb_list = []

    for t in range(len(interp_index)):
        # Find the positions and rotations between which we're interpolating
        start_idx = interp_index[t]
        end_idx = start_idx + 1
        start_pos = receiver_position[start_idx]
        end_pos = receiver_position[end_idx]

        start_rot = receiver_rotation_list[start_idx]
        end_rot = receiver_rotation_list[end_idx]

        # Interpolate position and rotation
        receiver_position_interp = interpolate_values(start_pos, end_pos, interp_weight[t])
        receiver_rotation_interp = interpolate_values(start_rot, end_rot, interp_weight[t])

        receiver = Receiver(receiver_position_interp, receiver_rotation_interp)
        scene.update_receiver(receiver)

        rgb, _ = scene.render_image()
        interpolated_rgb_list.append(rgb[..., :3])

    return interpolated_rgb_list
