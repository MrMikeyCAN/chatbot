import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, lemma
from model import NeuralNet
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

# Load intents from intents.json
with open("intents.json", "r") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Loop through each sentence in intents patterns
for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
