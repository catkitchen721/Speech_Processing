#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import pyaudio
import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt

sample_rate, dual_amp_data = wavfile.read('speech.wav') # 讀取聲音檔

amp_data = [] # 預備將雙聲道轉成單聲道
for u in dual_amp_data:
    amp_data.append(((u[0] - u[1]) // 2) + min(u[0], u[1])) # 雙聲道每個frame取平均得單聲道

print(sample_rate) # 取樣頻率

# 畫各個contour
fig, ax = plt.subplots(nrows=3, figsize=(12.0, 7.2))

# waveform
ax[0].plot(amp_data, '-b', linewidth=0.5)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[0].title.set_text('Waveform')
ax[0].grid(True)

# Energy
N = 450 # (for 44100Hz, 10.2ms)
energy_x = list(range(N//2, 244480, N))
energy_data = []
for u_x in energy_x:
    u_y = np.int64(0)
    for i in range(N):
        u_y += (amp_data[u_x - N//2 + i]) ** 2
    energy_data.append(u_y)
ax[1].plot(energy_x, energy_data, '-r', linewidth=0.5)
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Energy')
ax[1].title.set_text('Energy Contour (Window = 10.2ms)')
ax[1].grid(True)

# Zero-Crossing Rate
N = 450 # (for 44100Hz, 10.2ms)
zcr_x = list(range(N//2, 244480, N))
zcr_data = []
for u_x in zcr_x:
    u_y = 0
    for i in range(N-1):
        signal = [0, 0]
        if amp_data[u_x - N//2 + i] >= 0:
            signal[0] = 1
        else:
            signal[0] = -1
        
        if amp_data[u_x - N//2 + (i+1)] >= 0:
            signal[1] = 1
        else:
            signal[1] = -1
        u_y += (abs(signal[0] - signal[1]) // 2)
    zcr_data.append(u_y)
ax[2].plot(zcr_x, zcr_data, '-b', linewidth=0.5)
ax[2].set_xlabel('Time')
ax[2].set_ylabel('ZCR(times)')
ax[2].title.set_text('Zero-Crossing Rate Contour (Window = 10.2ms)')
ax[2].grid(True)

# show figure
plt.tight_layout()
plt.show()