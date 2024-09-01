import torch
from functools import lru_cache
import time


@lru_cache
def get_alphabet(file: str) -> str:
    with open(file, 'r', encoding='utf-8') as f:
        alphabet = f.read()
        if len(alphabet) != 96:
            raise ValueError('Alphabet size must be 96 characters')
        return alphabet


@lru_cache
def tokenize(text: str, alphabet: str,
             dtype: torch.dtype = torch.int8) -> torch.Tensor:
    tokenized_tensor = torch.tensor([alphabet.index(char)+1 for char in text], dtype=dtype)
    return tokenized_tensor
