import sounddevice as sd
import numpy as np
import wave
import matplotlib.pyplot as plt
import noise_filter as nf
from pedalboard import *

class Voicer:
    def __init__(self, format, channels, rate, duration):
        self.format = format
        self.channels = channels
        self.rate = rate
        self.frames = []
        self.duration = duration
        self.waveform = np.array([])

    def record(self):
        print("Recording for {} seconds...".format(self.duration))
        self.waveform = sd.rec(int(self.duration * self.rate), samplerate=self.rate, channels=self.channels, dtype=self.format)
        sd.wait()
        self.waveform = self.waveform.reshape(self.channels,-1)

    def save_wave(self, filename):
        wave_file = wave.open(filename, 'wb')
        wave_file.setnchannels(self.channels)
        wave_file.setsampwidth(self.audio.get_sample_size(self.format))
        wave_file.setframerate(self.rate)
        wave_file.writeframes(b''.join(self.frames))
        wave_file.close()

    def draw_waveform(self,waveform,rate,title="Waveform"):
        num_channels, num_frames = waveform.shape
        time_axis = np.arange(0, num_frames) / rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, self.waveform[c], linewidth=1)
            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c+1}")
        figure.suptitle(title)

    def draw_specgram(self,waveform,rate,title="Specgram"):
        num_channels,_ = waveform.shape

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].specgram(self.waveform[c], Fs=rate)
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c+1}")
        figure.suptitle(title)

voicer = Voicer("float32", 1, 16000,10)
voicer.record()

voicer.draw_specgram(voicer.waveform, rate=voicer.rate)
voicer.draw_waveform(voicer.waveform, rate=voicer.rate)

board = Pedalboard([
    NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
    Compressor(threshold_db=-16, ratio=2.5),
    LowShelfFilter(cutoff_frequency_hz=500, gain_db=10, q=1),
    Gain(gain_db=15)
])

audio_filter = nf.SoundFilter()
audio_filter.clear(board=board,audio=[voicer.waveform.flatten()])
audio_filter.playSounds()
plt.show()




