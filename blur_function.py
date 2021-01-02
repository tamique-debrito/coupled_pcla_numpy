import numpy as np
from scipy.signal import convolve2d
from utils import normalize_prob

WINDOW_OPTIONS = {'hann': np.hanning}


def make_window(win_size, win_padding, win_type):
    assert win_type in WINDOW_OPTIONS, f"Unsupported window type {win_type}"

    base = WINDOW_OPTIONS[win_type](win_size)
    return np.concatenate((base, np.zeros(win_padding)))


class BlurFn:
    def __init__(self, blur_filter, norm_axis=None):
        self.blur_filter = blur_filter
        self.norm_axis = norm_axis
    def __call__(self, spec):
        blurred = convolve2d(spec, self.blur_filter, mode='same')
        if self.norm_axis is not None:
            return normalize_prob(blurred, ax=self.norm_axis)
        else:
            return blurred

# Heuristic blur functions, as copying approach in the paper appeared to cause some artifacts in the filters
def make_blur_functions(win_size_t, win_size_f, fft_size, stride, win_type='hann', show_progress=False):
    """
    win_size_t < win_size_f
    """
    if show_progress:
        print("\rCreating blur functions", end='')
    win_diff = win_size_f - win_size_t
    strides_per_window = (win_size_f - win_size_t) // stride + 1  # how many times win_size_t can fit into win_size_f
    blur_t_size = strides_per_window

    blur_filter_t = np.hanning(blur_t_size)

    blur_f_size = int(np.ceil(fft_size / win_size_t * 8 + 1))

    blur_filter_f = np.hanning(blur_f_size)

    blur_filter_t = np.expand_dims(blur_filter_t, 1)
    blur_filter_f = np.expand_dims(blur_filter_f, 0)

    if show_progress:
        print("\rDone creating blur functions")

    return BlurFn(blur_filter_t, 0), \
           BlurFn(blur_filter_f, 1)


#  Attempt at copying approach in paper's matlab code for creating blur functions, but this seems to cause artifacts
#  This may be because of the difference between matlab's 'mldivide' and numpy's 'linalg.lstsq'
def _make_blur_functions(win_size_t, win_size_f, fft_size, stride, win_type='hann', show_progress=False):
    """
    win_size_t < win_size_f
    """
    if show_progress:
        print("\rCreating blur functions", end='')
    win_diff = win_size_f - win_size_t
    strides_per_window = (win_size_f - win_size_t) // stride + 1  # how many times win_size_t can fit into win_size_f
    blur_t_size = strides_per_window

    win_t = make_window(win_size_t, win_padding=win_diff, win_type=win_type)
    win_f = make_window(win_size_f, win_padding=0, win_type=win_type)

    overlapping_windows_t = np.empty((win_size_f, blur_t_size))
    for i in range(blur_t_size):
        overlapping_windows_t[:, i] = np.roll(win_t, stride * i)

    #  Finding the filter by approximately combining the filters sometimes creates incorrect filters, especially when t-blur size is large
    #  blur_filter_t = np.linalg.lstsq(overlapping_windows_t, win_f, rcond=None)[0]
    blur_filter_t = np.hanning(blur_t_size)

    win_t_fft = np.fft.fftshift(abs(np.fft.fft(win_t, n=fft_size)))
    win_f_fft = np.fft.fftshift(abs(np.fft.fft(win_f, n=fft_size)))

    blur_f_size = int(np.ceil(fft_size / win_size_t * 8 + 1))
    if blur_f_size % 2 == 0:
        blur_f_size += 1
    fft_offset = -(blur_f_size - 1) // 2

    overlapping_windows_f_fft = np.empty((fft_size, blur_f_size))
    for i in range(blur_f_size):
        overlapping_windows_f_fft[:, i] = np.roll(win_f_fft, fft_offset + i)

    blur_filter_f = np.linalg.lstsq(overlapping_windows_f_fft, win_t_fft, rcond=None)[0]

    blur_filter_t = np.expand_dims(blur_filter_t, 1)
    blur_filter_f = np.expand_dims(blur_filter_f, 0)

    if show_progress:
        print("\rDone creating blur functions")

    return BlurFn(blur_filter_t, 0), \
           BlurFn(blur_filter_f - blur_filter_f.min(), 1)
