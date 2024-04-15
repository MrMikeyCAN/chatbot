import audiofile
from rVADfast import rVADfast
import numpy as np


class VAD:
    def __init__(self):
        self.vad = rVADfast()
        self.audio = None
        self.sample_rate = None

    def load(self,filename: str):
        self.audio, self.sample_rate = audiofile.read(filename)

    def is_speech(self, waveform: np.ndarray = None, sample_rate: int = None):
        if waveform is not None and sample_rate is not None:
            self.audio = waveform
            self.sample_rate = sample_rate
        else:
            if self.audio is None:
                raise Exception("No audio file")
            elif self.sample_rate is None:
                raise Exception("No sample rate")

        vad_labels, vad_timestamps = self.vad(self.audio,self.sample_rate)
        return vad_labels, vad_timestamps

