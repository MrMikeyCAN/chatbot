from gtts import gTTS
from pygame import mixer
from nltk.corpus import wordnet
from transformers import AutoTokenizer
import pandas as pd

mixer.init()
mixer.music.set_volume(1)
import nltk
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

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize(text):
    tokenized_text = tokenizer(text=text,truncation=True, padding='max_length', max_length=42, return_tensors="pt")
    return tokenized_text["input_ids"]



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


class LanguageIndexMapper:
    def __init__(self, labels):
        self.label_to_index = {label: idx for idx, label in enumerate(set(labels))}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

    def label_to_index_func(self, label):
        return self.label_to_index[label]

    def index_to_label_func(self, index):
        return self.index_to_label[index]