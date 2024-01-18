from gtts import gTTS


def text_to_spech(bot_name: str, text: str, lang="en"):
    tts = gTTS(text=text, lang=lang)
    filename = "sound.waw"
    tts.save(filename)
    print(f"{bot_name}: {text}")
