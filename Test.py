from models.GPTModel import (
    Hyperparameters,
    TrainParameters,
    GPTLanguageModel,
    ModelFuncs,
)

from utils import save_to_csv

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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
    n_embd=16,
    n_head=6,
    n_layer=20,
    dropout=0.3,
    batch_size=16,
    block_size=16,
    decoder=decode,
    encoder=encode,
    device=device,
)

trainParams = TrainParameters(
    learning_rate=2e-4,
    device=device,
    max_iters=5000,
    checkpoint=100,
    decoder=decode,
    encoder=encode,
    eval_interval=1,
    eval_iters=100,
    visualate=False,
    text=text,
)

model = GPTLanguageModel(hyperParams).to(device)
modelFuncs = ModelFuncs(hyperparams=hyperParams, train_params=trainParams, model=model)
modelFuncs.m = modelFuncs.m.to(device)
print("Model cihazı:", next(model.parameters()).device)
print("Train params cihazı:", trainParams.device)
print("Hyper params cihazı:", hyperParams.device)
modelFuncs.train()
