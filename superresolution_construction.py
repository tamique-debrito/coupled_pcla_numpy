import numpy as np

def construct_superres_spectrogram(pf_z, pt_z, pf_fz, pt_tz):
    p_z = (pf_z + pt_z) / 2.0  # Using either would be fine, but just average to make implementation nicer (because of symmetry)

    p_z = np.expand_dims(p_z, axis=)
    pf_fz = np.expand_dims(p_f_fz, axis=)
    pt_tz = np.expand_dims(pt_tz, axis=)

    combined = p_z * pf_fz * pt_tz

    superres = np.sum(combined, axis=)

    return superres