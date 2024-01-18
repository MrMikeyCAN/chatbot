from gtts import gTTS
from pygame import mixer

mixer.init()


def text_to_spech(text: str, lang="en"):
    tts = gTTS(text=text, lang=lang)
    filename = "sound.mp3"
    tts.save(filename)
    mixer.music.load(filename)
    mixer.music.play()
