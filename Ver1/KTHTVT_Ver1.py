import numpy as np
from scipy.signal import hilbert
import sounddevice as sd
from scipy.io import savemat   # <-- [THÊM DÒNG NÀY 1]

# Thong so de tai
fs = 20000
t = np.arange(0, 1, 1/fs)
fc = 5000
f0 = 10

# Song mang
cos_c = np.cos(2 * np.pi * fc * t)
sin_c = np.sin(2 * np.pi * fc * t)

# 1. Song sin
x_sin = np.zeros(len(t))
for f in range(10, 101):
    pha = np.random.rand() * 2 * np.pi
    x_sin += np.cos(2 * np.pi * f * t + pha)
x_sin = x_sin / np.max(np.abs(x_sin))

# Hilbert 
h_sin = np.imag(hilbert(x_sin))
ssb_sin = x_sin * cos_c - h_sin * sin_c

# 2. Xung vuong 
x_sq = np.zeros(len(t))
for k in [1, 3, 5, 7, 9]:
    dau = (-1)**((k - 1) // 2)
    x_sq += (dau / k) * np.cos(2 * np.pi * k * f0 * t)
x_sq = (4 / np.pi) * x_sq

h_sq = np.imag(hilbert(x_sq))
ssb_sq = x_sq * cos_c - h_sq * sin_c

# 3. Tam giac
x_tri = np.zeros(len(t))
for k in [1, 3, 5, 7, 9]:
    x_tri += (1 / (k**2)) * np.cos(2 * np.pi * k * f0 * t)
x_tri = (8 / (np.pi**2)) * x_tri

h_tri = np.imag(hilbert(x_tri))
ssb_tri = x_tri * cos_c - h_tri * sin_c

# Chuan hoa bien do va ghep chuoi 15 giay
s1 = (ssb_sin / np.max(np.abs(ssb_sin))).astype(np.float32)
s2 = (ssb_sq / np.max(np.abs(ssb_sq))).astype(np.float32)
s3 = (ssb_tri / np.max(np.abs(ssb_tri))).astype(np.float32)

chuoi_tin_hieu = np.concatenate([np.tile(s1, 5), np.tile(s2, 5), np.tile(s3, 5)])

# Xuat file du lieu cho MATLAB
savemat('ssb_pi_data.mat', {'s1': s1, 's2': s2, 's3': s3, 'fs': fs}) # <-- [THÊM DÒNG NÀY 2]
print("-> Da xuat xong file ssb_pi_data.mat")

try:
    print("Phat song cho kit")
    sd.play(chuoi_tin_hieu, samplerate=fs, loop=True)
    sd.wait()
except KeyboardInterrupt:
    sd.stop()
    print("Dung phat tin hieu")