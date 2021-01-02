import numpy as np
from utils import normalize_prob
from initializations import init_params
from blur_function import make_blur_functions


def e_step(pt_z, pt_t_z, pf_z, pf_f_z, b_t, b_f):
    """
    :param pt_z: probability distribution of latent vectors for high-time-res plca; shape (n_latent,)
    :param pt_t_z: conditional probability distribution of temporal component for high-time-res plca,
                i.e. temporal basis; shape (n_latent, fft_size)
    :param pf_z: probability distribution of latent vectors for high-spec-res plca; shape (n_latent,)
    :param pf_f_z: conditional probability distribution of spectral component for high-spec-res plca,
                i.e. spectral basis; shape (n_latent, fft_size)
    :param b_t: blurring function for temporal basis
    :param b_f: blurring function for spectral basis
    :return: pt_z_tf, pf_z_tf
            where pt_z_tf conditional distribution of latent components for high-time-res plca;
                shape (n_latent, temporal_size, fft_size)
            and   pf_z_tf conditional distribution of latent components for high-spec-res plca
                shape (n_latent, temporal_size, fft_size)
    """

    pt_z_tf = normalize_prob(
        np.expand_dims(pt_z, (1, 2))
        * np.expand_dims(pt_t_z, 2)
        * np.expand_dims(b_f(pf_f_z), 1),
        0
    )
    pf_z_tf = normalize_prob(
        np.expand_dims(pf_z, (1, 2))
        * np.expand_dims(b_t(pt_t_z), 2)
        * np.expand_dims(pf_f_z, 1),
        0
    )

    return pt_z_tf, pf_z_tf


def m_step(pt_z_tf, spec_t, pf_z_tf, spec_f):
    """
    :param pt_z_tf: conditional distribution of latent components for high-time-res plca;
                shape (n_latent, temporal_size, fft_size)
    :param spec_t: high-time-res spectrogram; shape (temporal_size, fft_size)
    :param pf_z_tf: conditional distribution of latent components for high-spec-res plca;
                shape (n_latent, temporal_size, fft_size)
    :param spec_f:  high-spec-res spectrogram; shape (temporal_size, fft_size)
    :return: pt_z, pt_f_z, pf_z, pf_f_z
        pt_z: probability distribution of latent vectors for high-time-res plca; shape (n_latent,)
        pt_f_z: conditional probability distribution of spectral component for high-time-res plca,
                    i.e. spectral basis; shape (n_latent, fft_size)
        pt_t_z: conditional probability distribution of temporal component for high-time-res plca,
                    i.e. temporal basis; shape (n_latent, fft_size)
        pf_z: probability distribution of latent vectors for high-spec-res plca; shape (n_latent,)
    """
    weighted_density_t = np.expand_dims(spec_t, 0) * pt_z_tf
    weighted_density_f = np.expand_dims(spec_f, 0) * pf_z_tf

    pt_z = normalize_prob(np.sum(weighted_density_t, (1, 2)))
    pt_t_z = normalize_prob(np.sum(weighted_density_t, 2), 1)

    pf_z = normalize_prob(np.sum(weighted_density_f, (1, 2)))
    pf_f_z = normalize_prob(np.sum(weighted_density_f, 1), 1)

    return pt_z, pt_t_z, pf_z, pf_f_z


def run_coupled_plca(spec_t, spec_f, n_latent, n_iters, stft_options, use_spec_init=True, show_progress=False):
    temporal_size, fft_size, win_size_t, win_size_f, stride = stft_options
    pt_z, pt_t_z, pf_z, pf_f_z = init_params(n_latent, temporal_size, fft_size,
                                             spec_f=spec_f if use_spec_init else None,
                                             show_progress=show_progress)
    b_t, b_f = make_blur_functions(win_size_t, win_size_f, fft_size, stride, show_progress=show_progress)

    for i in range(n_iters):
        if show_progress:
            print(f"\rRunning coupled PLCA, iteration {i+1}/{n_iters}", end='')
        pt_z_tf, pf_z_tf = e_step(pt_z, pt_t_z, pf_z, pf_f_z, b_t, b_f)
        pt_z, pt_t_z, pf_z, pf_f_z = m_step(pt_z_tf, spec_t, pf_z_tf, spec_f)

        assert all([x.min() >= 0 for x in [pt_z_tf, pf_z_tf, pt_z, pt_t_z, pf_z, pf_f_z]]), 'a pmf has negative probabilities'
    if show_progress:
        print("\rDone running coupled PLCA")

    return pt_z, pt_t_z, pt_z_tf, pf_z, pf_f_z, pf_z_tf
