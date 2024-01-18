import numpy as np
import nltk

from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.probability import FreqDist
from nltk import bigrams
from nltk import ConditionalFreqDist


lemmatizer = WordNetLemmatizer()

nltk.download("stopwords")

nltk.download("punkt")
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def tokenize(sentence):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(sentence)
    tokens = [
        word.lower()
        for word in words
        if word.isalnum() and word.lower() not in stop_words
    ]
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


# Sample text for demonstration purposes
sample_text = "This is a sample sentence. Tokenization is a crucial step in natural language processing."


# Tokenization logic


def tokenize_text(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase for case-insensitivity
    return tokens


# Initialization logic
def initialize_ngram_model(tokens, n=2):
    ngrams = list(
        bigrams(tokens)
    )  # Using bigrams for simplicity, you can adjust n for higher order n-grams
    cfd = ConditionalFreqDist((ngram[:-1], ngram[-1]) for ngram in ngrams)
    return cfd


# Example usage
tokens = tokenize_text(sample_text)
ngram_model = initialize_ngram_model(tokens)

# Now, you can use ngram_model to predict the next word based on the context.
# For example, if you want to predict the next word after "sample", you can do:
context = ("sample",)
predicted_next_word = ngram_model[context].max()
print(f"Predicted next word: {predicted_next_word}")
