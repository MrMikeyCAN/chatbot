import numpy as np
import nltk

from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")


def tokenize(word):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(word)
    tokens = [word.lower() for word in words if word.lower() not in stop_words]
    return tokens


def lemma(word):
    """
    lemmatization = find the base form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [lemma(w) for w in words]
    -> ["organize", "organize", "organize"]
    """
    if isinstance(word, list):
        return [lemmatizer.lemmatize(w.lower()) for w in word]
    else:
        return lemmatizer.lemmatize(word.lower())


import numpy as np


def bag_of_words(tokenized_sentence, words):
    # lemmatize each word
    sentence_words = [lemma(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag
