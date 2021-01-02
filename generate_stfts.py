from scipy.io.wavfile import read
from scipy.signal import stft

def gen_stfts(source_path, win_size_t, win_size_f, fft_size, stride, max_temporal_size=None, show_progress=False, start=0):
    audio = read(source_path)[1][start:]

    # align_shift = (win_size_f - win_size_t) // 2
    # audio_t_shifted = audio[align_shift:]

    if show_progress:
        print("\rGenerating high-time-res spectrogram", end='')
    spec_t = stft(audio, nperseg=win_size_t, nfft=fft_size, noverlap=win_size_t - stride)[2].transpose()
    if show_progress:
        print("\rDone generating high-time-res spectrogram")
        print("\rGenerating high-spec-res spectrogram", end='')
    spec_f = stft(audio, nperseg=win_size_f, nfft=fft_size, noverlap=win_size_f - stride)[2].transpose()
    if show_progress:
        print("\rDone generating high-spec-res spectrogram")

    common_len = min(spec_t.shape[0], spec_f.shape[0])
    if max_temporal_size is not None:
        common_len = min(common_len, max_temporal_size)

    spec_t = spec_t[:common_len]
    spec_f = spec_f[:common_len]

    return abs(spec_t), abs(spec_f)