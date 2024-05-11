import torch

from Test import hyperParams, trainParams
from models.GPTModel import ModelFuncs

model = torch.load("chechpoints/checkpoint:3400.pkl")

model_funcs = ModelFuncs(model=model, hyperparams, trainParams)

print(model_funcs.Generate_Text(context="my name is", max_new_tokens=10))
