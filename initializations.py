import numpy as np
import random
from utils import normalize_prob, epsilon

"""
Parameters:
pt_z = PT(z)
pt_t_z = PT(t|z)
pt_z_ft = PT(z|f,t)

pf_z = PF(z)
pf_f_z = PF(f|z)
pf_z_ft = PF(z|f,t)

"""


def random_indices(n, k):
    return random.sample([i for i in range(n)], k)


def init_params(n_latent, temporal_size, fft_size, spec_f=None, show_progress=False):
    """
    Initializes probability functions for pcla
    if 'VF' is not None will intitialize the spectral weights using VF

    :returns pt_t_z, pt_z, pf_f_z, pf_z
            pt_t_z is temporal basis for high-temporal-res; shape (n_latent, temporal_size)
            pt_z is latent distribution for high-temporal-res; shape (n_latent)
            pf_t_z is temporal basis for high-spectral-res; shape (n_latent, temporal_size)
            pf_z is latent distribution for high-spectral-res; shape (n_latent)
    """
    if show_progress:
        print("\rInitializing parameters", end='')

    pt_z = np.random.random(n_latent) + 1
    pt_z = pt_z / np.sum(pt_z)
    pt_t_z = np.random.random((n_latent, temporal_size)) + 1
    pt_t_z = normalize_prob(pt_t_z, ax=0)

    pf_z = np.random.random(n_latent) + 1
    pf_z = pf_z / np.sum(pf_z)
    if spec_f is not None:
        n = spec_f.shape[0]
        pf_f_z = spec_f[random_indices(n, n_latent)] + epsilon
    else:
        pf_f_z = np.random.random((n_latent, fft_size)) + 1
    pf_f_z = normalize_prob(pf_f_z, ax=0)
    np.random.random((n_latent, fft_size)) + 1

    if show_progress:
        print("\rDone initializing parameters")

    return pt_z, pt_t_z, pf_z, pf_f_z

