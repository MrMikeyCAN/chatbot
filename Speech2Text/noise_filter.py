from df.enhance import enhance, init_df
import numpy as np
import torchaudio
import torch


class NoiseFilter:
    def __init__(self, version: str = "DeepFilterNet3"):
        self.model, self.df_state, self._ = init_df(model_base_dir=version)
        self.audio = None
        self.sample_rate = None
        self.default_sr = self.df_state.sr()

    def noise_filter(self, audio: np.array = None, sample_rate: int = None):
        if audio is not None and sample_rate is not None:
            self.audio = torch.from_numpy(audio)
            self.sample_rate = sample_rate
        else:
            if self.audio is None:
                raise ValueError("Noise filter needs audio input")
            elif self.sample_rate is None:
                raise ValueError("Noise filter needs sample rate")

        if self.sample_rate is not self.default_sr:
            self.resample(self.default_sr)

        enhanced = enhance(self.model, self.df_state, self.audio)
        return enhanced

    def save(self, filename: str, audio: torch.tensor):
        torchaudio.save(filename, audio, self.sample_rate)

    def load(self, filename: str, normalize: bool = True):
        self.audio, self.sample_rate = torchaudio.load(filename, normalize=normalize)

    def resample(self, new_sr: int, audio: np.ndarray = None, sample_rate: int = None):
        if audio is not None and sample_rate is not None:
            self.audio = torch.from_numpy(audio)
            self.sample_rate = sample_rate
        else:
            if self.audio is None:
                raise ValueError("Noise filter needs audio input")
            elif self.sample_rate is None:
                raise ValueError("Noise filter needs sample rate")

        transform = torchaudio.transforms.Resample(self.sample_rate, new_sr)
        self.audio = transform(self.audio)
        self.sample_rate = new_sr
