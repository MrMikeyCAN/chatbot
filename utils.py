from gtts import gTTS


def text_to_spech(text: str, lang="en"):
    tts = gTTS(text=text, lang=lang)
    filename = "sound.mp3"
    tts.save(filename)
