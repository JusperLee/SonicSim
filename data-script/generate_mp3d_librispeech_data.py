import os
import json
import argparse
import itertools
import subprocess
import typing as T
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import imageio
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import *
import gc
from ss_utils import Receiver, Source, Scene
from rich import print
import time

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

import habitat_sim.sim
import ss_utils
import habitat_utils
import audio_utils
import dynamic_utils
import tool_utils
import multiprocessing
import logging

def process_single(scene, sample_rate, novel_path_config, channel_type, results_dir, source1_path, source2_path, noise_path, music_path, transcripts):
    """
    Save:
    ├── {results_demo} = LibriSpeech-MP3D/{mode}/{room}/{spk1-id}-{spk2-id}
    │   ├── video/
    │   │   ├── moving_audio.wav    : Audio interpolated for the moving receiver.
    │   │   ├── moving_audio_1.wav  : Audio interpolated specifically for source 1.
    │   │   ├── moving_audio_2.wav  : Audio interpolated specifically for source 2.
    │   │   ├── moving_video.mp4    : Video visualization of movement (no audio).
    │   │   ├── nvas.mp4            : NVAS video results with combined audio.
    │   │   ├── nvas_source1.mp4    : NVAS video results for only source 1 audio.
    │   │   ├── nvas_source2.mp4    : NVAS video results for only source 2 audio.
    │   │   └── rgb_receiver.png    : A rendered view from the perspective of the receiver.
    """
    # Constants
    sample_rate = sample_rate
    novel_path_config = novel_path_config
    channel_type = channel_type
    
    spks_nav_points = [ss_utils.get_nav_idx(scene, distance_threshold=6.0) for _ in range(2)]
    spks_nav_mid_points = [spks_nav_points[i][len(spks_nav_points[i]) // 2] for i in range(2)]
    mic_points = ss_utils.get_nav_point_from_grid_points(scene, spks_nav_mid_points, distance_threshold=6.0, num_points=1)[0]
    noise_music_points = ss_utils.get_nav_point_from_grid_points(scene, spks_nav_mid_points, distance_threshold=6.0, num_points=2)
    print(mic_points)
    print(noise_music_points)
    grid_points = [spks_nav_points, mic_points, noise_music_points]
    ss_utils.save_trace_gif(scene, "./trace.png", grid_points)

    # Generate RIRs
    output_dir = f'{results_dir}'
    os.makedirs(output_dir, exist_ok=True)
    ir_save_dir = f'{output_dir}/ir_save_{novel_path_config}_{channel_type}.pt'
    
    # Merge spks_nav_points
    merge_spks_nav_points = []
    for i in range(len(spks_nav_points)):
        merge_spks_nav_points += spks_nav_points[i]
    
    ir_outputs = []
    for i in range(len(spks_nav_points)):
        ir_output = audio_utils.generate_rir_combination(
                room, spks_nav_points[i], [mic_points], [90], channel_type
        )
        ir_outputs.append(ir_output)
        del ir_output
        gc.collect()
        
    torch.save(ir_outputs, ir_save_dir)
    # import pdb; pdb.set_trace()
    
    ir1_list, ir2_list = ir_outputs
    
    source1_audio, start_end_points1, audioname1 = audio_utils.create_long_audio(source1_path, 60)
    source2_audio, start_end_points2, audioname2 = audio_utils.create_long_audio(source2_path, 60)
    
    # Interpolate audio for moving receiver
    receiver_audio_1 = dynamic_utils.interpolate_moving_audio(source1_audio, ir1_list, spks_nav_points[0])
    receiver_audio_2 = dynamic_utils.interpolate_moving_audio(source2_audio, ir2_list, spks_nav_points[1])
    
    
    # Get noise and music audio
    noise_audio, noise_start_end, noise_audioname = audio_utils.create_background_audio(noise_path, 60)
    music_audio, music_start_end, music_audioname = audio_utils.create_background_audio(music_path, 60)
    
    # Get rir for noise and music
    rir_noise = ss_utils.render_ir(room, noise_music_points[0], mic_points, filename=None, receiver_rotation=90, channel_type=channel_type, channel_order=0)
    rir_music = ss_utils.render_ir(room, noise_music_points[1], mic_points, filename=None, receiver_rotation=90, channel_type=channel_type, channel_order=0)
    # import pdb; pdb.set_trace()
    rir_noise = torch.from_numpy(dynamic_utils.convolve_fixed_receiver(noise_audio, rir_noise))
    rir_music = torch.from_numpy(dynamic_utils.convolve_fixed_receiver(music_audio, rir_music))
    # import pdb; pdb.set_trace()
    # Save audio
    receiver_audio_1 = audio_utils.get_lufs_norm_audio(receiver_audio_1.transpose(0,1).numpy(), sample_rate, -17)[0]
    receiver_audio_2 = audio_utils.get_lufs_norm_audio(receiver_audio_2.transpose(0,1).numpy(), sample_rate, -17)[0]
    rir_noise = audio_utils.get_lufs_norm_audio(rir_noise.transpose(0,1).numpy(), sample_rate, -24)[0]
    rir_music = audio_utils.get_lufs_norm_audio(rir_music.transpose(0,1).numpy(), sample_rate, -29)[0]
    torchaudio.save(f'{output_dir}/moving_audio_1.wav', torch.from_numpy(receiver_audio_1).transpose(0,1), sample_rate=sample_rate)
    torchaudio.save(f'{output_dir}/moving_audio_2.wav', torch.from_numpy(receiver_audio_2).transpose(0,1), sample_rate=sample_rate)
    torchaudio.save(f'{output_dir}/noise_audio.wav', torch.from_numpy(rir_noise).transpose(0,1), sample_rate=sample_rate)
    torchaudio.save(f'{output_dir}/music_audio.wav', torch.from_numpy(rir_music).transpose(0,1), sample_rate=sample_rate)
    
    ss_utils.save_trace_gif(scene, f"{output_dir}/trace.png", grid_points)
    
    
    json_dicts = {
        'source1': {
            'audio': audioname1,
            'start_end_points': start_end_points1,
            'words': [transcripts[name] for name in audioname1]
        },
        'source2': {
            'audio': audioname2,
            'start_end_points': start_end_points2,
            'words': [transcripts[name] for name in audioname2]
        },
        'noise': {
            'audio': noise_audioname,
            'start_end_points': noise_start_end
        },
        'music': {
            'audio': music_audioname,
            'start_end_points': music_start_end
        },
    }
    
    with open(f'{output_dir}/json_data.json', 'w') as f:
        logging.info(json_dicts)
        json.dump(json_dicts, f)

def removing_exist_speaker(root, speech_lists):
    exist_folders = os.listdir(root)
    exist_speakers = []
    for folder in exist_folders:
        exist_speakers.append(folder.split("-")[0])
        exist_speakers.append(folder.split("-")[1])
    exist_speakers = list(set(exist_speakers))
    new_speech_lists = []
    for speech in speech_lists:
        if speech.split("/")[-1] not in exist_speakers:
            new_speech_lists.append(speech)
    return new_speech_lists

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(f'generate_mp3d_librispeech_data_{args.idx}.log'), logging.StreamHandler()])
    
    multiprocessing.set_start_method("spawn")  # 使用spqwn模式
    channel_type = 'Binaural'
    sample_rate = 16000
    transcripts = tool_utils.load_transcripts('librispeech/train-360.csv')
    for mode in ["train"]:
        with open(f"data/{mode}_scene.txt", "r") as f:
            scene_list = f.readlines()
        scene_list = [scene.strip() for scene in scene_list]
        logging.info(f"Training on {len(scene_list)} scenes")
        for idx, scene in enumerate(scene_list):
            total_time = 0.0
            logging.info(f"Processing {mode} {idx}/{len(scene_list)} {scene}")
            
            with open(f"data/{mode}_speech.txt", "r") as f:
                speech_list = f.readlines()
            
            speech_list = [speech.strip() for speech in speech_list]
            
            if os.path.exists(f"LibriSpace/{mode}/{scene}"):
                speech_list = removing_exist_speaker(f"LibriSpace/{mode}/{scene}", speech_list)
                logging.info(f"Removing, {len(speech_list)} exist speakers")
                
            # Extract and load room and grid related data
            room = scene
            loaded_scene = ss_utils.Scene(
                room,
                [None],  # placeholder for source class
                include_visual_sensor=False,
                device=torch.device('cpu')
            )
            
            while len(speech_list) > 2:
                start_time = time.time()
                selected_speech = np.random.choice(speech_list, 2, replace=False)
                speech_list = [speech for speech in speech_list if speech not in selected_speech]
                source1_path = selected_speech[0]
                source2_path = selected_speech[1]
                
                noise_path = f"data/{mode}_noise.json"
                music_path = f"data/{mode}_music.json"
                
                results_dir = f'LibriSpace/{mode}/{scene}/{source1_path.split("/")[-1].split(".")[0]}-{source2_path.split("/")[-1].split(".")[0]}'
                os.makedirs(results_dir, exist_ok=True)
                logging.info(f"Processing {mode} {idx}/{len(scene_list)} {scene} {source1_path.split('/')[-1].split('.')[0]}-{source2_path.split('/')[-1].split('.')[0]}")
                
                process_single(loaded_scene, sample_rate, mode, channel_type, results_dir, source1_path, source2_path, noise_path, music_path, transcripts)
                end_time = time.time()
                logging.info(f"Time elapsed: {(end_time - start_time)/60} min, Length of speech list: {len(speech_list)}")
                total_time += (end_time - start_time)/60
                # break
            logging.info("Total time: {} min".format(total_time))
                
                
            