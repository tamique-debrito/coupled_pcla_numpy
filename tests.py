from blur_function import *
from generate_stfts import *
from superresolution_construction import *

from utils import normalize_prob

import matplotlib.pyplot as plt

TEST_WIN_T_SIZE = 16
TEST_WIN_F_SIZE = 64
TEST_STRIDE = 2
TEST_FFT_SIZE = 1024


def gen_spectrograms_for_testing():
    return gen_stfts('jobs.wav', TEST_WIN_T_SIZE, TEST_WIN_F_SIZE,
                     TEST_FFT_SIZE, TEST_STRIDE, max_temporal_size=2000)


def test_gen_stfts():
    spec_t, spec_f = gen_spectrograms_for_testing()

    plt.subplot(1, 2, 1)
    plt.imshow(spec_t)
    plt.title('spect-t')
    plt.subplot(1, 2, 2)
    plt.imshow(spec_f)
    plt.title('spec-f')
    plt.show()

def test_blur_functions_lines():

    b_t, b_f = make_blur_functions(TEST_WIN_T_SIZE, TEST_WIN_F_SIZE, TEST_FFT_SIZE, TEST_STRIDE, win_type='hann')

    lines = np.zeros((256, 256))
    lines[0::16] = 1
    lines[1::16] = 1
    lines[2::16] = 1
    lines[:, 0::16] = 1
    lines[:, 1::16] = 1
    lines[:, 2::16] = 1

    # show blur filters
    plt.subplot(2, 1, 1)
    plt.plot(b_t.blur_filter.squeeze())
    plt.title('filter-t')
    plt.subplot(2, 1, 2)
    plt.plot(b_f.blur_filter.squeeze())
    plt.title('filter-f')
    plt.show()

    # Show blurred images
    blurred_t = b_t(lines)
    blurred_f = b_f(lines)
    plt.subplot(1, 3, 1)
    plt.imshow(lines)
    plt.title('original')
    plt.subplot(1, 3, 2)
    plt.imshow(blurred_t)
    plt.title('blur t')
    plt.subplot(1, 3, 3)
    plt.imshow(blurred_f)
    plt.title('blur f')
    plt.show()


def test_blur_functions_spectrograms():
    spec_t, spec_f = gen_spectrograms_for_testing()
    spec_t = normalize_prob(spec_t, 0)
    spec_f = normalize_prob(spec_f, 1)

    b_t, b_f = make_blur_functions(TEST_WIN_T_SIZE, TEST_WIN_F_SIZE, TEST_FFT_SIZE, TEST_STRIDE, win_type='hann')

    # show blur filters
    plt.subplot(2, 1, 1)
    plt.plot(b_t.blur_filter.squeeze())
    plt.title('filter-t')
    plt.subplot(2, 1, 2)
    plt.plot(b_f.blur_filter.squeeze())
    plt.title('filter-f')
    plt.show()

    # Show blurred images
    blurred_t = b_t(spec_t)
    blurred_f = b_f(spec_f)
    plt.subplot(2, 2, 1)
    plt.imshow(spec_t[:250])
    plt.title('original t')
    plt.subplot(2, 2, 2)
    plt.imshow(spec_f[:250])
    plt.title('original norm_f')
    plt.subplot(2, 2, 3)
    plt.imshow(blurred_t[:250])
    plt.title('blur-time')
    plt.subplot(2, 2, 4)
    plt.imshow(blurred_f[:250])
    plt.title('blur-spec')
    plt.show()


if __name__ == "__main__":
    test_gen_stfts()
    test_blur_functions_spectrograms()
