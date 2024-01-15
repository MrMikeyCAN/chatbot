import numpy as np
import nltk

nltk.download("punkt")
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)


def lemma(word):
    """
    lemmatization = find the base form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [lemma(w) for w in words]
    -> ["organize", "organize", "organize"]
    """
    return lemmatizer.lemmatize(word.lower())


def bag_of_words(tokenized_sentence, words):
    # lemmatize each word
    sentence_words = [lemma(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag
