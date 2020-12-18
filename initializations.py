import numpy as np
import random
"""
Parameters:
pf_z = PF(z)
pf_f_z = PF(f|z)
pf_t_z = PF(t|z)
pf_z_ft = PF(z|f,t)

pt_z = PT(z)
pt_f_z = PT(f|z)
pt_t_z = PT(t|z)
pt_z_ft = PT(z|f,t)

"""


epsilon = 1e-10

def random_indices(n, k):
    return random.sample([i for i in range(n)], k)

def init_params(n_basis, spec_f=None):
    """
    if 'spec_f' is not None will intitialize the spectral weights using spec_f
    """
    if spec_f is not None:
        n = spec_f.shape[1]
        pf_z = spec_f[:, random_indices(n, n_basis)]
    else:
        pf_z = np.random.random(())