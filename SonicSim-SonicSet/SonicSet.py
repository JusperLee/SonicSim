import os
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import torchaudio
import numpy as np
import gc
from rich import print
import time

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import habitat_sim.sim
import SonicSim_rir
import SonicSim_habitat
import SonicSim_audio
import SonicSim_moving
import tool_utils
import torch.multiprocessing as mp
import logging

def process_single(scene, sample_rate, novel_path_config, channel_type, mic_array_list, results_dir, source1_path, source2_path, source3_path, noise_path, music_path, transcripts):
    # Constants
    sample_rate = sample_rate
    novel_path_config = novel_path_config
    channel_type = channel_type
    
    # Extract and load room and grid related data
    room = scene
    scene = SonicSim_rir.Scene(
        room,
        [None],  # placeholder for source class
        include_visual_sensor=False,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )
    
    spks_nav_points = [SonicSim_rir.get_nav_idx(scene, distance_threshold=5.0) for _ in range(3)]
    spks_nav_mid_points = [spks_nav_points[i][len(spks_nav_points[i]) // 2] for i in range(3)]
    mic_points = SonicSim_rir.get_nav_point_from_grid_points(scene, spks_nav_mid_points, distance_threshold=6.0, num_points=1)[0]
    noise_music_points = SonicSim_rir.get_nav_point_from_grid_points(scene, spks_nav_mid_points, distance_threshold=6.0, num_points=2)
    print(mic_points)
    print(noise_music_points)
    grid_points = [spks_nav_points, mic_points, noise_music_points]
    SonicSim_rir.save_trace_gif(scene, "./trace.png", grid_points)
    scene.sim.close()
    # Generate RIRs
    output_dir = f'{results_dir}'
    os.makedirs(output_dir, exist_ok=True)
    ir_save_dir = f'{output_dir}/rir_save_{novel_path_config}_{channel_type}.pt'
    
    # Merge spks_nav_points
    merge_spks_nav_points = []
    for i in range(len(spks_nav_points)):
        merge_spks_nav_points += spks_nav_points[i]
    
    ir_outputs = []
    for i in range(len(spks_nav_points)):
        ir_output = SonicSim_audio.generate_rir_combination(
                room, spks_nav_points[i], [mic_points], [90], mic_array_list, channel_type
        )
        ir_outputs.append(ir_output.cpu())
        del ir_output
        gc.collect()
        
    torch.save(ir_outputs, ir_save_dir)
    
    ir1_list, ir2_list, ir3_list = ir_outputs
    
    source1_audio, start_end_points1, audioname1 = SonicSim_audio.create_long_audio(source1_path, 60)
    source2_audio, start_end_points2, audioname2 = SonicSim_audio.create_long_audio(source2_path, 60)
    source3_audio, start_end_points3, audioname3 = SonicSim_audio.create_long_audio(source3_path, 60)
    
    # Interpolate audio for moving receiver
    receiver_audio_1 = SonicSim_moving.interpolate_moving_audio(source1_audio, ir1_list, spks_nav_points[0])
    receiver_audio_2 = SonicSim_moving.interpolate_moving_audio(source2_audio, ir2_list, spks_nav_points[1])
    receiver_audio_3 = SonicSim_moving.interpolate_moving_audio(source3_audio, ir3_list, spks_nav_points[2])
    
    # Get noise and music audio
    noise_audio, noise_start_end, noise_audioname = SonicSim_audio.create_background_audio(noise_path, 60)
    music_audio, music_start_end, music_audioname = SonicSim_audio.create_background_audio(music_path, 60)
    
    # Get rir for noise and music
    if channel_type == 'CustomArrayIR':
        rir_noise = SonicSim_rir.create_custom_arrayir(room, noise_music_points[0], mic_points, mic_array=mic_array_list, filename=None, receiver_rotation=90, channel_order=0)
        rir_music = SonicSim_rir.create_custom_arrayir(room, noise_music_points[1], mic_points, mic_array=mic_array_list, filename=None, receiver_rotation=90, channel_order=0)
    else:
        rir_noise = SonicSim_rir.render_ir(room, noise_music_points[0], mic_points, filename=None, receiver_rotation=90, channel_type=channel_type, channel_order=0)
        rir_music = SonicSim_rir.render_ir(room, noise_music_points[1], mic_points, filename=None, receiver_rotation=90, channel_type=channel_type, channel_order=0)

    rir_noise = torch.from_numpy(SonicSim_moving.convolve_fixed_receiver(noise_audio, rir_noise.cpu()))
    rir_music = torch.from_numpy(SonicSim_moving.convolve_fixed_receiver(music_audio, rir_music.cpu()))

    # Save audio
    receiver_audio_1 = SonicSim_audio.get_lufs_norm_audio(receiver_audio_1.transpose(0,1).numpy(), sample_rate, -17)[0]
    receiver_audio_2 = SonicSim_audio.get_lufs_norm_audio(receiver_audio_2.transpose(0,1).numpy(), sample_rate, -17)[0]
    receiver_audio_3 = SonicSim_audio.get_lufs_norm_audio(receiver_audio_3.transpose(0,1).numpy(), sample_rate, -17)[0]
    rir_noise = SonicSim_audio.get_lufs_norm_audio(rir_noise.transpose(0,1).numpy(), sample_rate, -24)[0]
    rir_music = SonicSim_audio.get_lufs_norm_audio(rir_music.transpose(0,1).numpy(), sample_rate, -29)[0]
    torchaudio.save(f'{output_dir}/moving_audio_1.wav', torch.from_numpy(receiver_audio_1).transpose(0,1), sample_rate=sample_rate)
    torchaudio.save(f'{output_dir}/moving_audio_2.wav', torch.from_numpy(receiver_audio_2).transpose(0,1), sample_rate=sample_rate)
    torchaudio.save(f'{output_dir}/moving_audio_3.wav', torch.from_numpy(receiver_audio_3).transpose(0,1), sample_rate=sample_rate)
    torchaudio.save(f'{output_dir}/noise_audio.wav', torch.from_numpy(rir_noise).transpose(0,1), sample_rate=sample_rate)
    torchaudio.save(f'{output_dir}/music_audio.wav', torch.from_numpy(rir_music).transpose(0,1), sample_rate=sample_rate)
    
    # SonicSim_rir.save_trace_gif(scene, f"{output_dir}/trace.png", grid_points)
    
    json_dicts = {
        'source1': {
            'audio': audioname1,
            'start_end_points': start_end_points1,
            'words': [transcripts[os.path.basename(name)] for name in audioname1]
        },
        'source2': {
            'audio': audioname2,
            'start_end_points': start_end_points2,
            'words': [transcripts[os.path.basename(name)] for name in audioname2]
        },
        'source3': {
            'audio': audioname3,
            'start_end_points': start_end_points3,
            'words': [transcripts[os.path.basename(name)] for name in audioname3]
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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(f'SonicSet-train.log'), logging.StreamHandler()])
    
    mp.set_start_method('spawn', force=True)
    channel_type = 'Mono' # Supported: 'Ambisonics', 'Binaural', 'Mono', 'CustomArrayIR'
    mic_array_list = None
    # For CustomArrayIR
    """
    channel_type = 'CustomArrayIR'
    mic_array_list = [
        [0, 0, 0],
        [0, 0, 0.04],
        [0, 0, 0.12],
        [0, 0, 0.16]
    ] # 4-channel linear microphone array
    """
    """
    channel_type = 'CustomArrayIR'
    mic_array_list = [
        [0, 0, -0.035],
        [0.035, 0, 0],
        [0, 0, 0.035],
        [-0.035, 0, 0]
    ] # 4-channel circular microphone array
    """
    sample_rate = 16000
    for mode in ["train", "val", "test"]:
        transcripts = tool_utils.load_transcripts(f'LibriSpeech/{mode}.csv')
        with open(f"SonicSet/demo-split-json/{mode}_scene.txt", "r") as f:
            scene_list = f.readlines()
        scene_list = [scene.strip() for scene in scene_list]
        logging.info(f"Training on {len(scene_list)} scenes")
        for idx, scene in enumerate(scene_list):
            total_time = 0.0
            logging.info(f"Processing {mode} {idx}/{len(scene_list)} {scene}")
            
            with open(f"SonicSet/demo-split-json/{mode}_speech.txt", "r") as f:
                speech_list = f.readlines()
            
            speech_list = [speech.strip() for speech in speech_list]
            
            if os.path.exists(f"SonicSet/scene_datasets/mp3d/{mode}/{scene}"):
                speech_list = removing_exist_speaker(f"SonicSet/scene_datasets/mp3d/{mode}/{scene}", speech_list)
                logging.info(f"Removing, {len(speech_list)} exist speakers")
                
            while len(speech_list) >= 3:
                start_time = time.time()
                selected_speech = np.random.choice(speech_list, 3, replace=False)
                speech_list = [speech for speech in speech_list if speech not in selected_speech]
                source1_path = selected_speech[0]
                source2_path = selected_speech[1]
                source3_path = selected_speech[2]
                
                noise_path = f"SonicSet/demo-split-json/{mode}_noise.json"
                music_path = f"SonicSet/demo-split-json/{mode}_music.json"
                
                results_dir = f'SonicSet/scene_datasets/mp3d/{mode}/{scene}/{source1_path.split("/")[-1].split(".")[0]}-{source2_path.split("/")[-1].split(".")[0]}-{source3_path.split("/")[-1].split(".")[0]}'
                os.makedirs(results_dir, exist_ok=True)
                logging.info(f"Processing {mode} {idx}/{len(scene_list)} {scene} {source1_path.split('/')[-1].split('.')[0]}-{source2_path.split('/')[-1].split('.')[0]}-{source3_path.split('/')[-1].split('.')[0]}")
                
                process_single(scene, sample_rate, mode, channel_type, mic_array_list, results_dir, source1_path, source2_path, source3_path, noise_path, music_path, transcripts)
                end_time = time.time()
                logging.info(f"Time elapsed: {(end_time - start_time)/60} min, Length of speech list: {len(speech_list)}")
                total_time += (end_time - start_time)/60
                torch.cuda.empty_cache()
                gc.collect()
            logging.info("Total time: {} min".format(total_time))
                
                
            