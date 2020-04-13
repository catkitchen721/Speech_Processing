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
fig, ax = plt.subplots(nrows=4, figsize=(12.0, 8.0))

# waveform
ax[0].plot(amp_data, '-b', linewidth=0.5)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[0].title.set_text('Waveform')
ax[0].grid(True)

# Energy
N = 450 # (for 44100Hz, 10.2ms)
energy_x = list(range(N//2, len(amp_data) - N, N))
energy_data = []
for u_x in energy_x:
    u_y = np.int64(0)
    for i in range(N):
        u_y += (amp_data[u_x - N//2 + i]) ** 2
    energy_data.append(u_y)
ax[1].plot(energy_x, energy_data, '-r', linewidth=0.5, label='energy')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Energy')
ax[1].title.set_text('Energy Contour (Window = 10.2ms), with end-points')
ax[1].grid(True)

# Zero-Crossing Rate
N = 450 # (for 44100Hz, 10.2ms)
zcr_x = list(range(N//2, len(amp_data) - N, N))
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

# End Point Detection
N = 450 # (for 44100Hz, 10.2ms)
ignore_threshold = 4500
ITU = 1 * 1e10
ITL = 4e9
IZCT = 30
is_begun = False
is_ended = True
points = [] # end points

for i in range(len(energy_data)): # first scan
    if energy_x[i] > ignore_threshold:
        if is_ended: # 尋找起點
            if energy_data[i] > ITU:
                points.append(energy_x[i])
                is_begun = True
                is_ended = False
        if is_begun: # 尋找終點
            if energy_data[i] < ITU:
                points.append(energy_x[i])
                is_begun = False
                is_ended = True

points2 = [points[0]]
for i in range(len(points)): # remove bugs
    if (i != 0) and (i != (len(points) - 1)) and (i % 2 == 1) :
        if abs(points[i] - points[i+1]) > 5 * N:
            points2.append(points[i])
            points2.append(points[i+1])
points2.append(points[-1])

'''
for i in range(len(points2) - 1): # find real points
    if (i % 2 == 0): # backward
        for j in range(len(energy_x)):
            if energy_x[j] == points2[i]:
                for k in range(8):
                    if energy_data[j-k-1] < ITL:
                        points2[i] = energy_x[j-k-1]
    else: # forward
        for j in range(len(energy_x)):
            if energy_x[j] == points2[i]:
                for k in range(8):
                    if energy_data[j+k+1] < ITL:
                        points2[i] = energy_x[j+k+1]
'''
print("End-Point:")
print(points2)
points2_y = np.zeros(len(points2))
ax[1].plot(points2, points2_y, '.k', linewidth=2, label='end_point')
ax[1].legend(loc="upper right")

# Pitch 
short_time = N / sample_rate # 0.0102s = 10.2ms
pitch_x = []
pitch_data = []

zcr_dict = {}
for i in range(len(zcr_x)):
    zcr_dict[zcr_x[i]] = i
# print(zcr_dict)

i = 0
while i < len(points2):
    start = zcr_dict[points2[i]]
    end = zcr_dict[points2[i+1]]
    for j in range(len(zcr_x)):
        if j >= start and j <= end:
            freq = int(zcr_data[j] / short_time) # 算頻率
            if freq < 3000:
                pitch_x.append(zcr_x[j])
                pitch_data.append(freq)
    i += 2
# print(pitch_x)
# print(pitch_data)
ax[3].plot(pitch_x, pitch_data, '.k', linewidth=1)
ax[3].set_xlabel('Time')
ax[3].set_ylabel('Pitch Contour')
ax[3].title.set_text('Pitch Contour (Window = 10.2ms)')
ax[3].grid(True)

# show figure
plt.tight_layout()
plt.show()