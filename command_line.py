import argparse

def gen_argparser():
    parser = argparse.ArgumentParser(description='Run coupled PLCA')

    # Audio source
    parser.add_argument('--source', metavar='SOURCE', type=str,
                        help='source audio file', default='test.wav')

    # STFT Options
    parser.add_argument('--win_t', metavar='WIN_T', type=int,
                        help='short window size (for high temporal resolution)', default=64)
    parser.add_argument('--win_f', metavar='WIN_F', type=int,
                        help='long window size (for high spectral resolution)', default=512)
    parser.add_argument('--stride', metavar='STRIDE', type=int,
                        help='STFT stride (hop) size', default=16)
    parser.add_argument('--fft_size', metavar='FFT_SIZE', type=int,
                        help='fft size', default=1024)

    # PLCA Options
    parser.add_argument('--n_latent', metavar='N_LATENT', type=int,
                        help='number of latent variables for PLCA', default=50)
    parser.add_argument('--n_iters', metavar='N_ITERS', type=int,
                        help='number of iterations for PLCA', default=10)
    parser.add_argument('--spec_init', metavar='SPEC_INIT', type=bool,
                        help='whether to initialize spectral basis with samples from true spectrum', default=True)

    # Misc
    parser.add_argument('--show_prog', metavar='SHOW_PROG', type=bool,
                        help='whether to show progress', default=False)



    return parser