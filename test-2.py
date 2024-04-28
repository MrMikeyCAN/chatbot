from models.GPTModel import hyperparams, GPTLanguageModel, Generate_Text,encode
import torch

model = GPTLanguageModel(hyperparams)
model.load_state_dict(torch.load("chechpoints/checkpoint:3400.pkl"))


context = "my name is"

# Modeli oluştururken aynı parametreleri kullanmalısınız
print(Generate_Text(context=context, max_new_tokens=10))
