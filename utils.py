from gtts import gTTS
from pygame import mixer
from nltk_utils import lemma, tokenize

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


ignore_words = [
    ",",
    "!",
    "?",
    "*",
    "-",
    "#",
    "(",
    ")",
    ".",
    "<",
    "'",
    "...",
    "--",
    "{",
    "}",
    "~",
]


def text_to_indices(text, vocab):
    return [
        vocab[word] if word in vocab else 0
        for word in tokenize(text)
        if word not in ignore_words
    ]


# Assuming you have a function to build a vocabulary and a tokenizer
def build_vocab(texts):
    all_words = [lemma(w) for text_tokens in texts for w in text_tokens if w not in ignore_words]
    vocab = {word: idx + 1 for idx, word in enumerate(sorted(set(all_words)))}
    return vocab