import torch
import numpy as np
import os
import pandas as pd

def find_matching_indices(tensor_list, array_list):
    matching_indices = []
    # 对 tensor_list 中的每个元素
    for i, tensor in enumerate(tensor_list):
        # 对 array_list 中的每个元素
        for j, array in enumerate(array_list):
            # 比较 tensor_array 和 array 是否相等
            if torch.norm(tensor - torch.from_numpy(array)) < 0.5:
                # 保存 tensor_list 中的索引
                matching_indices.append(i)  # 存储 (tensor_list索引, array_list索引)
    return list(set(matching_indices))  # 去重

def process_librispeech(root_dir, save_dir):
    records = []

    # 遍历所有子文件夹和文件
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".txt"):  # 找到转录文本文件
                text_path = os.path.join(subdir, file)
                with open(text_path, 'r') as text_file:
                    for line in text_file:
                        parts = line.strip().split()
                        audio_file_name = parts[0] + ".flac"  # 假设音频文件是flac格式
                        transcript = ' '.join(parts[1:])
                        records.append([audio_file_name, transcript])

    # 将结果保存到CSV文件
    df = pd.DataFrame(records, columns=['name', 'words'])
    df.to_csv(save_dir, index=False)

def load_transcripts(csv_file_path):
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 将DataFrame转换为字典，其中文件名是键，转录文本是值
    transcripts_dict = pd.Series(df.words.values,index=df.name).to_dict()

    return transcripts_dict