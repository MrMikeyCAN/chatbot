# from gtts import gTTS
# from pygame import mixer
# from datasets import load_dataset, Audio
# from transformers import AutoFeatureExtractor

# mixer.init()
# mixer.music.set_volume(1)

# def text_to_speech(bot_name: str, text: str, lang="tr"):
#     tts = gTTS(text=text, lang=lang)
#     filename = "sound.wav"
#     tts.save(filename)
#     print(f"{bot_name}: {text}")
#     mixer.music.load(filename)
#     mixer.music.play()
#     while mixer.music.get_busy():
#         continue

import whisper
import sounddevice as sd
import numpy as np
import wave
from pedalboard import Pedalboard, NoiseGate, Compressor, LowShelfFilter, Gain

def record_audio(duration, rate=16000):
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * rate), samplerate=rate, channels=1, dtype='float32')
    sd.wait()
    return recording.flatten()

def apply_effects(audio, rate=16000):
    board = Pedalboard([
        NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
        Compressor(threshold_db=-16, ratio=2.5),
        LowShelfFilter(cutoff_frequency_hz=500, gain_db=10, q=1),
        Gain(gain_db=15)
    ])
    return board(audio, rate)

def save_wave(filename, audio, rate=16000):
    with wave.open(filename, 'wb') as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)
        wave_file.setframerate(rate)
        wave_file.writeframes((audio * 32767).astype(np.int16).tobytes())

duration = 10  # 5 seconds of audio
audio_data = record_audio(duration)
processed_audio = apply_effects(audio_data)
save_wave('processed_audio.wav', processed_audio)
print("Recording finished and saved to 'processed_audio.wav'")

model = whisper.load_model("base")
result = model.transcribe("clear.wav")
print(result["text"])
