import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# 创建一个大图
fig = plt.figure(figsize=(25, 5))


y, sr = librosa.load("files/61-908-7127/noise_audio.wav")

# 计算音频的短时傅里叶变换 (Short-Time Fourier Transform)
D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)

# 绘制频谱图
librosa.display.specshow(D, sr=sr, x_axis='time', cmap='inferno')

# 调整子图间距
plt.savefig("Figure_2.png", dpi=600)
# 显示图像
# plt.show()