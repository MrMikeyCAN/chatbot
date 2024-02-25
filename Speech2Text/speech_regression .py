import pyaudio
import numpy as np
import matplotlib.pyplot as plt
  
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5

audio = pyaudio.PyAudio()
  
# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
print ("recording...")
frames = []
  
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(np.frombuffer(data, dtype=np.int16))
print ("finished recording")

# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()

def plot_waveform(waveform, sample_rate):
    num_channels, num_frames = waveform.shape
    time_axis = np.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")

waveform = np.concatenate(frames)
waveform = waveform.reshape(CHANNELS,-1)

print(waveform.shape)
plot_waveform(waveform, RATE)
plt.show()