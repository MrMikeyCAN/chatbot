import torch
import torchaudio
import matplotlib.pyplot as plt

filename = "Datasets\\tr\\clips\\common_voice_tr_17341269.mp3"

def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)

waveform, sample_rate = torchaudio.load(filename)

plot_specgram(waveform, sample_rate)