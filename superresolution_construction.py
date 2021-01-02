import numpy as np

def construct_superres_spectrogram(pt_z, pt_t_z, pf_z, pf_f_z, show_progress):
    if show_progress:
        print("\rConstructing superresolution spectrogram", end='')
    p_z = (pt_z + pf_z) / 2.0  # Using either would be fine, but just average to make implementation nicer (because of symmetry)

    p_z = np.expand_dims(p_z, axis=(1, 2))
    pt_t_z = np.expand_dims(pt_t_z, axis=2)
    pf_f_z = np.expand_dims(pf_f_z, axis=1)

    combined = p_z * pt_t_z * pf_f_z

    superres = np.sum(combined, axis=0)

    if show_progress:
        print("\rDone constructing superresolution spectrogram")

    return superres