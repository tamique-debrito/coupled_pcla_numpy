import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

epsilon = 1e-10

def init_distribution(shape, ax=0):
    arr = np.random.random(shape) + 1
    return normalize_prob(arr, ax)


def normalize_prob(arr, ax=0):
    return arr / np.expand_dims(np.sum(arr, axis=ax), axis=ax)


def show_spec(spec, name='spectrogram'):
    spec = abs(spec).transpose()
    spec = np.log(spec + epsilon)
    plt.figure(figsize=(5, 10))
    plt.imshow(spec)
    plt.title(name)
    plt.show()


def pad(x, n):
    r_pad = n // 2
    l_pad = n - r_pad
    if n > 0:
        return np.concatenate((np.zeros(l_pad), x, np.zeros(r_pad)))
    else:
        return x[-l_pad:r_pad]


def get_samp(x, win_size, stride, fft_size, t):
    samp = x[t * stride:t * stride + win_size]
    pad_size = fft_size - win_size
    samp = pad(samp, pad_size)
    samp = samp * np.hanning(samp.size)
    return samp


def alt_stft(x, win_size, fft_size, stride):
    t_size = (x.size - win_size) // stride + 1
    spec = np.empty((t_size, fft_size))
    for t in range(t_size):
        samp = get_samp(x, win_size, fft_size, stride, t)
        spec[t] = abs(fft(samp, fft_size))
    return spec