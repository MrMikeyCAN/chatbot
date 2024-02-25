import sounddevice as sd
from scipy.io.wavfile import write
import torch
from transformers import pipeline

"""def record_audio(duration=5, fs=16000, filename='output.wav'):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Kaydın bitmesini bekleyin
    write(filename, fs, recording)
    print(f"Recording saved as {filename}")

# Ses kaydını yap
record_audio()
"""

# Pipeline için ayarlar"
model_id = "openai/whisper-large"

# Whisper modeli için pipeline oluştur
whisper_pipe = pipeline(model=model_id, task="automatic-speech-recognition")

# Kaydedilmiş ses dosyasını işle
result = whisper_pipe("temp_audio (1).wav")
print(result["text"])
