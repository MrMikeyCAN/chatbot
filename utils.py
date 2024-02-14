from gtts import gTTS
from pygame import mixer
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor

mixer.init()
mixer.music.set_volume(1)

def text_to_speech(bot_name: str, text: str, lang="tr"):
    tts = gTTS(text=text, lang=lang)
    filename = "sound.wav"
    tts.save(filename)
    print(f"{bot_name}: {text}")
    mixer.music.load(filename)
    mixer.music.play()
    while mixer.music.get_busy():
        continue