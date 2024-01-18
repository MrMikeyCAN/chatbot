from gtts import gTTS
from pygame import mixer

mixer.init()
mixer.music.set_volume(1)

def text_to_speech(bot_name: str, text: str, lang="en"):
    tts = gTTS(text=text, lang=lang)
    filename = "sound.mp3"
    tts.save(filename)
    print(f"{bot_name}: {text}")
    mixer.music.load(filename)
    mixer.music.play()
    while mixer.music.get_busy():
        continue
