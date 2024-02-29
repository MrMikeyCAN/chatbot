"""import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from pedalboard import Pedalboard, NoiseGate, Compressor, LowShelfFilter, Gain
import noise_filter as nf

class VoicerSD:
    def __init__(self, channels, rate, duration):
        self.channels = channels
        self.rate = rate
        self.duration = duration
        self.frames = []
        self.waveform = np.array([])

    def record(self):
        print("Recording for {} seconds...".format(self.duration))
        self.waveform = sd.rec(int(self.duration * self.rate), samplerate=self.rate, channels=self.channels, dtype='float32')
        sd.wait()  # Wait until the recording is finished

    def draw_waveform(self, title="Waveform"):
        plt.figure(figsize=(10, 4))
        plt.plot(np.linspace(0, self.duration, num=len(self.waveform)), self.waveform)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)

    def draw_specgram(self, title="Spectrogram"):
        plt.figure(figsize=(10, 4))
        plt.specgram(self.waveform.flatten(), Fs=self.rate)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")

# Create a VoicerSD instance
voicer = VoicerSD(channels=1, rate=16000, duration=10)  # 10 seconds duration
voicer.record()

# Draw waveform and spectrogram
voicer.draw_waveform()
voicer.draw_specgram()

# Setup the pedalboard
board = Pedalboard([
    NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
    Compressor(threshold_db=-16, ratio=2.5),
    LowShelfFilter(cutoff_frequency_hz=500, gain_db=10, q=1),
    Gain(gain_db=15)
])

audio_filter = nf.SoundFilter()
audio_filter.clear(board=board,audio=[voicer.waveform.flatten()])
audio_filter.playSounds()

plt.show()"""
