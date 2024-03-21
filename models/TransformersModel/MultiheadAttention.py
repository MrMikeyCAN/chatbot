import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# * Bu cümlenin kaç kelime içerdiğini gösterecektir
sequence_lenght = 4
batch_size = 1
input_dim = 512
d_model = 512
X = torch.randn((batch_size, sequence_lenght, input_dim))

# print(X.size())

qkv_layer = nn.Linear(input_dim, 3 * d_model)

qkv = qkv_layer(X)
# print(qkv.shape)

# TODO Öngörülen boyut
num_heads = 8

# TODO modeli boyutuna bölüyoruz tüm boyutları tek tek inceliyoruz
head_dim = d_model // num_heads

# TODO qkv değerlerini ayırmaya uygun hale getiriyoruz
qkv = qkv.reshape(batch_size, sequence_lenght, num_heads, 3 * head_dim)

# print(qkv.shape)
# TODO qkv değerlerine değerlerini atıyoruz
qkv = qkv.permute(0, 2, 1, 3)
# print(qkv.shape)


# TODO qkv değerlerini parçalıyoruz
q, k, v = qkv.chunk(3, dim=-1)
# print(q.shape, k.shape, v.shape)


d_k = q.size()[-1]
scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
print(scaled.shape)

# TODO Üssünü alma işlemi
k.mT.shape

y = torch.randn(2, 3)
torch.transpose(y, 0, 1)


torch.transpose(y, 1, 0)


k.transpose(-1, -2) == k.transpose(-2, -1)


k.transpose(-1, -2).shape

# TODO Maskeleme işlemi olası işlem hatalarını(özellikle de softmax içerisinde yaşanabilecekleri) engellemek için yapılır
mask = torch.full(scaled.size(), float("-inf"))
mask = torch.triu(mask, diagonal=1)
mask[0][1]  # mask for input to a single head

(scaled + mask)[0][0]

# print(np.exp(0.5596) / (np.exp(0.5596) + np.exp(0.0404)))

# todo softmax fonksiyounu
attention = F.softmax(scaled, dim=-1)


# print(attention.shape)

# print(attention[0][0])

# todo Ekstra fonksiyonlarımızı oluşturuyoruz


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


values, attention = scaled_dot_product(q, k, v, mask=mask)


# print(attention.shape)

# print(attention[0][0])

# print(values.size())


values = values.reshape(batch_size, sequence_lenght, num_heads * head_dim)
# print(values.size())

linear_layer = nn.Linear(d_model, d_model)

out = linear_layer(values)

# print(out.shape)

# print(out)


# todo Attention mekanizması için ekstra fonksiyon
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(input_dim, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, sequence_length, input_dim = x.size()
        print(f"x.size(): {x.size()}")
        qkv = self.qkv_layer(x)
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.reshape(
            batch_size, sequence_length, self.num_heads, 3 * self.head_dim
        )
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.permute(0, 2, 1, 3)
        print(f"qkv.size(): {qkv.size()}")
        q, k, v = qkv.chunk(3, dim=-1)
        print(f"q size: {q.size()}, k size: {k.size()}, v size: {v.size()}, ")
        values, attention = scaled_dot_product(q, k, v, mask)
        print(f"values.size(): {values.size()}, attention.size:{ attention.size()} ")
        values = values.reshape(
            batch_size, sequence_length, self.num_heads * self.head_dim
        )
        print(f"values.size(): {values.size()}")
        out = self.linear_layer(values)
        print(f"out.size(): {out.size()}")
        return out


input_dim = 1024
d_model = 512
num_heads = 8

batch_size = 30
sequence_length = 5
x = torch.randn((batch_size, sequence_length, input_dim))

model = MultiheadAttention(input_dim, d_model, num_heads)
out = model.forward(x)
