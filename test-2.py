from models.GPTModel import hyperparams, GPTLanguageModel, decode, encode
import torch

model = GPTLanguageModel(hyperparams)
model.load_state_dict(torch.load("chechpoint/checkpoint:0.pkl"))


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
m = model.to(device)


def Generate_Text(
    context: str,
    max_new_tokens: int = 500,
):
    context = torch.tensor(encode(context), device=device)[None, :]
    print(decode(m.generate(context, max_new_tokens)[0].tolist()))


# Modeli oluştururken aynı parametreleri kullanmalısınız
Generate_Text("Hi I am mert")
