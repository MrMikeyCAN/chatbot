import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# * Bu cümlenin kaç kelime içerdiğini gösterecektir
sequence_lenght = 4
batch_size = 1
input_dim = 512
d_model = 512
X = torch.randn((batch_size, sequence_lenght, input_dim))

print(X.size())

qkv_layer = nn.Linear(input_dim, 3 * d_model)
