import torchaudio
import matplotlib.pyplot as plt

# Ses dosyasını yükle
waveform, sample_rate = torchaudio.load('Datasets/tr/clips/common_voice_tr_17341269.mp3')

# Spektrogramı hesapla
spectrogram_transform = torchaudio.transforms.Spectrogram()
spectrogram = spectrogram_transform(waveform)

# Spektrogramı görselleştir
plt.figure()
plt.imshow(spectrogram.log2()[0, :, :].numpy(), cmap='viridis')
plt.colorbar()
plt.title("Spektrogram")
plt.xlabel("Time Bins")
plt.ylabel("Frequency Bins")
plt.show()
