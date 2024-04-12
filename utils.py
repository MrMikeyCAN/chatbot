from gtts import gTTS
from pygame import mixer
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor

mixer.init()
mixer.music.set_volume(1)

def text_to_speech(bot_name: str, text: str, lang="en"):
    tts = gTTS(text=text, lang=lang)
    filename = "sound.waw"
    tts.save(filename)
    print(f"{bot_name}: {text}")
    mixer.music.load(filename)
    mixer.music.play()
    while mixer.music.get_busy():
        continue

from pydub import AudioSegment

sound1 = AudioSegment.from_file("sound.wav")
sound2 = AudioSegment.from_file("combined_speech_crowd.wav")-20

combined = sound1.overlay(sound2)

combined.export("combined.wav", format='wav')