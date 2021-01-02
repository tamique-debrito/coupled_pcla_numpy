# Summary
This is a python implementation (using numpy and scipy) of the coupled Probabilistic Component Analysis (PLCA) for super-resolution spectrograms.  
Coupled PCLA is introduced the paper https://ccrma.stanford.edu/~juhan/pubs/jnam-interspeech10.pdf

## Some notes:

For STFT parameters, usual argument order is: window_size, fft_size, stride.  
For probabilistic variables, the usual argument order is time-related before frequency-related  .
For PCLA, the variables are named in format pA_B_C,
where A in ['t', 'f'] describes which of the high-temporal-res or high-spectral-res cases it belongs to;
B and C represent the conditioning of the probability (i.e. B|C).


### The probabilities modelled (assuming a joint distribution between spectrogram entries and latent vectors):

pt_z = PT(z)  
pt_t_z = PT(t|z)  
pt_z_ft = PT(z|f,t)  

pf_z = PF(z)  
pf_f_z = PF(f|z)  
pf_z_ft = PF(z|f,t)  

(PT represents the distribution associated with the high-temporal-resolution spectrogram
and PF is the distribution associated with the high-spectral-resolution spectrogram).

Note that PT(f|z) and PF(t|z) are not explicitly modelled,
as it is assumed (per the idea of coupling)
that PT(f|z) can be reconstructed by blurring PF(f|z) and similarly for PF(t|z).
