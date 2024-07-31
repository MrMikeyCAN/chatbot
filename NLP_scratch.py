from time import perf_counter
# import nltk
#
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag, word_tokenize, RegexpParser


def tokenizing_text_data(file_path: str):
    t = perf_counter()
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    print(perf_counter() - t)
    return tags


tokenizing_text_data("input.txt")
