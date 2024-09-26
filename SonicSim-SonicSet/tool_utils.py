import torch
import numpy as np
import os
import pandas as pd

def find_matching_indices(tensor_list, array_list):
    matching_indices = []
    for i, tensor in enumerate(tensor_list):
        for j, array in enumerate(array_list):
            if torch.norm(tensor - torch.from_numpy(array)) < 0.5:
                matching_indices.append(i)
    return list(set(matching_indices))

def process_librispeech(root_dir, save_dir):
    records = []

    # 遍历所有子文件夹和文件
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".txt"): 
                text_path = os.path.join(subdir, file)
                with open(text_path, 'r') as text_file:
                    for line in text_file:
                        parts = line.strip().split()
                        audio_file_name = parts[0] + ".flac" 
                        transcript = ' '.join(parts[1:])
                        records.append([audio_file_name, transcript])

    df = pd.DataFrame(records, columns=['name', 'words'])
    df.to_csv(save_dir, index=False)

def load_transcripts(csv_file_path):
    df = pd.read_csv(csv_file_path)
    transcripts_dict = pd.Series(df.words.values,index=df.name).to_dict()

    return transcripts_dict