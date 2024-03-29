import torch
import numpy as np

# TODO Basit bir tokenizer örneği

# Limit Number of sentences
TOTAL_SENTENCES = 100000


max_sequence_length = 200


def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True


def is_valid_length(sentence, max_sequence_length):
    return len(list(sentence)) < (
        max_sequence_length - 1
    )  # need to re-add the end token so leaving 1 space
