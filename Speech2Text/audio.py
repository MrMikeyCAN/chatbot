import torch
import torchaudio
from functools import lru_cache
import time

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)


@lru_cache
def load_audio(audio_path: str,
               target_sr: int = SAMPLE_RATE,
               stereo_split: bool = False) -> torch.Tensor:
    waveform, current_sr = torchaudio.load(audio_path, normalize=True)

    if current_sr != target_sr:
        waveform = torchaudio.functional.resample(waveform,
                                                  orig_freq=current_sr,
                                                  new_freq=target_sr)
    if stereo_split:
        return waveform[0], waveform[1]

    return waveform[0]


@lru_cache
def log_mel_spectrogram(waveform, mel_spectrogram,
                        dtype: torch.dtype = torch.float16) -> torch.Tensor:
    log_mel_spectrogram = mel_spectrogram(waveform)
    log_mel_spectrogram = torch.log(log_mel_spectrogram + 1e-9)
    log_mel_spectrogram = log_mel_spectrogram.to(dtype=dtype)
    return log_mel_spectrogram.T
