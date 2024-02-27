import pyaudio
import numpy as np
import webrtcvad
import wave
import matplotlib.pyplot as plt
import noise_filter as nf
from pedalboard import *

class VoiceActivityDetector:
    def __init__(self, format, channels, rate, chunk_duration_ms,silence_duration_ms = 500,voice_agression=3):
        self.vad = webrtcvad.Vad(voice_agression)
        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(rate * chunk_duration_ms / 1000)
        self.silence_duration_ms = silence_duration_ms
        self.silence_frames_threshold = int(self.silence_duration_ms / chunk_duration_ms)
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=format, channels=channels,
                                      rate=rate, input=True,
                                      frames_per_buffer=self.chunk_size)
        self.frames = []
        self.silence_frames_counter = 0
        self.waveform = np.array([])

    def record(self):
        print("Recording...")
        while True:
            data = self.stream.read(self.chunk_size)
            is_speech = self.vad.is_speech(data, self.rate)
            
            if is_speech:
                print("Speech detected!")
                self.silence_frames_counter = 0
            else:
                print("Silence.")
                self.silence_frames_counter += 1
            
            self.frames.append(np.frombuffer(data, dtype=np.int16))

            if self.silence_frames_counter >= self.silence_frames_threshold:
                break
        self.waveform = np.concatenate(self.frames)
        self.waveform = self.waveform.reshape(self.channels,-1)

    def save_wave(self, filename):
        wave_file = wave.open(filename, 'wb')
        wave_file.setnchannels(self.channels)
        wave_file.setsampwidth(self.audio.get_sample_size(self.format))
        wave_file.setframerate(self.rate)
        wave_file.writeframes(b''.join(self.frames))
        wave_file.close()

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def draw_waveform(self,title="Waveform"):
        num_channels, num_frames = self.waveform.shape
        time_axis = np.arange(0, num_frames) / self.rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, self.waveform[c], linewidth=1)
            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c+1}")
        figure.suptitle(title)

    def draw_specgram(self,title="Specgram"):
        num_channels,_ = self.waveform.shape

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].specgram(self.waveform[c], Fs=self.rate)
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c+1}")
        figure.suptitle(title)

vad = VoiceActivityDetector(pyaudio.paInt16, 1, 16000, 10,1000,2)
vad.record()
vad.save_wave("out.wav")
vad.close()

vad.draw_specgram()
vad.draw_waveform()

board = Pedalboard([
    NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
    Compressor(threshold_db=-16, ratio=2.5),
    LowShelfFilter(cutoff_frequency_hz=400, gain_db=10, q=1),
    Gain(gain_db=10)
])

audio_filter = nf.SoundFilter()
audio_filter.read("out.wav")
audio_filter.clear(board=board)
audio_filter.write("clear.wav")
audio_filter.playSounds()
plt.show()




