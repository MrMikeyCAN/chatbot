from gtts import gTTS
from pygame import mixer
from nltk.corpus import wordnet

mixer.init()
mixer.music.set_volume(1)

import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")


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


def tokenize(text):
    if isinstance(text, list):
        return text
    else:
        return word_tokenize(text)


def lemma(word):
    if isinstance(word, list):
        return [
            lemmatizer.lemmatize(w.lower()) for w in word if word not in ignore_words
        ]
    else:
        return lemmatizer.lemmatize(w.lower() for w in word if word not in ignore_words)


def bag_of_words(tokenized_sentence, words):
    sentence_words = [lemma(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag


def learn_meaning(word: str):
    synsets = wordnet.synsets(word)
    meanings = [syn.definition() for syn in synsets]
    return meanings


def text_to_speech(bot_name: str, text: str, lang="en"):
    tts = gTTS(text=text, lang=lang)
    filename = "sound.mp3"
    tts.save(filename)
    print(f"{bot_name}: {text}")
    mixer.music.load(filename)
    mixer.music.play()
    while mixer.music.get_busy():
        continue
