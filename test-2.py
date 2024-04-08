from models.TransformerModel import Transformer
from Test import create_masks, translate, transformer
import torch
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Modeli oluştururken aynı parametreleri kullanmalısınız
transformer.load_state_dict(torch.load("model_weights.pkl"))


translation = translate("hi i am mert")
print(translation)

translation = translate("hello how are you")
print(translation)
