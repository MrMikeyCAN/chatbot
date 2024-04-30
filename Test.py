from models.GPTModel import (
    Hyperparameters,
    TrainParameters,
    GPTLanguageModel,
    ModelFuncs,
)

import torch

with open("input2.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split the text into words

words = text.split()
vocab_size = len(set(words))
# Create a mapping from words to integers
stoi = {w: i for i, w in enumerate(set(words))}
itos = {i: w for i, w in enumerate(set(words))}


def encode(sentence):
    return [stoi[word] for word in sentence.split()]


def decode(indices):
    return " ".join(itos[i] for i in indices)


# Define the hyperparameters
hyperParams = Hyperparameters(
    vocab_size=vocab_size,
    n_embd=256,
    n_head=6,
    n_layer=20,
    dropout=0.3,
    batch_size=128,
    block_size=128,
    decoder=decode,
    encoder=encode,
)

trainParams = TrainParameters(
    learning_rate=2e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_iters=5000,
    checkpoint=100,
    decoder=decode,
    encoder=encode,
    eval_interval=1,
    eval_iters=100,
    visualate=False,
    text=text,
)

model = GPTLanguageModel(hyperParams)

modelFuncs = ModelFuncs(hyperparams=hyperParams, train_params=trainParams, model=model)

modelFuncs.train()
