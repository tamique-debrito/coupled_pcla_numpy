from generate_stfts import gen_stfts
from coupled_pcla import run_coupled_plca
from superresolution_construction import construct_superres_spectrogram
from command_line import gen_argparser
from utils import show_spec

# Set to true if it is preferrable to run this script using command line
USE_COMMAND_LINE = False

SOURCE_AUDIO_PATH = 'jobs.wav'

# STFT Options
WIN_T_SIZE = 64
WIN_F_SIZE = 512
STRIDE = 4
FFT_SIZE = 1024

# PLCA Options
N_LATENT = 150
N_ITERS = 50
USE_SPEC_INIT = True

SHOW_PROGRESS = True


def run_main_procedure(source_path, win_size_t, win_size_f, fft_size, stride,
                       n_latent, n_iters, use_spec_init,
                       show_progress):
    spec_t, spec_f = gen_stfts(source_path, win_size_t, win_size_f, fft_size, stride,
                               show_progress=show_progress, max_temporal_size=2000, start=2000)
    temporal_size = spec_t.shape[0]
    stft_options = (temporal_size, fft_size, win_size_t, win_size_f, stride)

    pt_z, pt_t_z, pt_z_tf, pf_z, pf_f_z, pf_z_tf = run_coupled_plca(spec_t, spec_f,
                                                                    n_latent, n_iters, stft_options, use_spec_init,
                                                                    show_progress)

    super_res = construct_superres_spectrogram(pt_z, pt_t_z, pf_z, pf_f_z, show_progress)

    return spec_t, spec_f, super_res


if __name__ == "__main__":
    if USE_COMMAND_LINE:
        parser = gen_argparser()
        args = parser.parse_args()
        source_path = args.source_path
        win_size_t, win_size_f, fft_size, stride = args.win_t, args.win_f, args.fft_size, args.stride
        n_latent, n_iters, use_spec_init = args.n_latent, args.n_iters, args.spec_init
        show_progress = args.show_prog
    else:
        source_path = SOURCE_AUDIO_PATH
        win_size_t, win_size_f, fft_size, stride = WIN_T_SIZE, WIN_F_SIZE, FFT_SIZE, STRIDE
        n_latent, n_iters, use_spec_init = N_LATENT, N_ITERS, USE_SPEC_INIT
        show_progress = SHOW_PROGRESS

    spec_t, spec_f, super_res = run_main_procedure(source_path, win_size_t, win_size_f, fft_size, stride,
                                   n_latent, n_iters, use_spec_init,
                                   show_progress)

    show_spec(spec_t, 'time')
    show_spec(spec_f, 'spec')
    show_spec(super_res, 'super')

