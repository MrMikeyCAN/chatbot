import torch
import math
from torch import nn
import torch.nn.functional as F

# ! Yüksek oranda encoder'a benziyor ancak bir nebze daha gelişmiş hali böylelikle artık sonuçlara varmamız ve daha verimli sonuçlar alabilmemiz sağlanıyor

# ! Hem decoder'da hem de encoder'da her işlemden sonra nelerin değiştiğini takip edebilmemiz için print kodları çalışmakta


def scaled_dot_product(q, k, v, mask=None):
    # q: 30 x 8 x 200 x 64, k: 30 x 8 x 200 x 64, v: 30 x 8 x 200 x 64, mask 200 x 200
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)  # 30 x 8 x 200 x 200
    print(f"scaled.size() : {scaled.size()}")
    if mask is not None:
        print(f"-- ADDING MASK of shape {mask.size()} --")
        scaled += mask  # 30 x 8 x 200 x 200
    attention = F.softmax(scaled, dim=-1)  # 30 x 8 x 200 x 200
    values = torch.matmul(attention, v)  # 30 x 8 x 200 x 64
    return values, attention


# TODO Burada da attention mekanizmaları sonrası olasılıkları öncesine ve sonrasına göre bulmasını sağlıyor
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        #  x: 30 x 200 x 512
        x = self.linear1(x)  # 30 x 200 x 2048
        print(f"x after first linear layer: {x.size()}")
        x = self.relu(x)  # 30 x 200 x 2048
        print(f"x after relu layer: {x.size()}")
        x = self.dropout(x)  # 30 x 200 x 2048
        print(f"x after dropout layer: {x.size()}")
        x = self.linear2(x)  # 30 x 200 x 512
        print(f"x after 2nd linear layer: {x.size()}")
        return x  # 30 x 200 x 512


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))  # 512
        self.beta = nn.Parameter(torch.zeros(parameters_shape))  # 512

    def forward(self, inputs):
        # inputs : 30 x 200 x 512
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]  # [-1]
        print(f"dims: {dims}")
        mean = inputs.mean(dim=dims, keepdim=True)  # 30 x 200 x 1
        print(f"Mean ({mean.size()})")
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)  # 30 x 200 x 512
        std = (var + self.eps).sqrt()  # 30 x 200 x 512
        print(f"Standard Deviation  ({std.size()})")
        y = (inputs - mean) / std  # 30 x 200 x 512
        print(f"y: {y.size()}")
        out = self.gamma * y + self.beta  # 30 x 200 x 512
        print(f"out: {out.size()}")
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)  # 1536
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, sequence_length, d_model = x.size()  # 30 x 200 x 512
        print(f"x.size(): {x.size()}")
        qkv = self.qkv_layer(x)  # 30 x 200 x 1536
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.reshape(
            batch_size, sequence_length, self.num_heads, 3 * self.head_dim
        )  # 30 x 200 x 8 x 192
        print(f"qkv after reshape .size(): {qkv.size()}")
        qkv = qkv.permute(0, 2, 1, 3)  # 30 x 8 x 200 x 192
        print(f"qkv after permutation: {qkv.size()}")
        q, k, v = qkv.chunk(
            3, dim=-1
        )  # q: 30 x 8 x 200 x 64, k: 30 x 8 x 200 x 64, v: 30 x 8 x 200 x 64
        print(f"q: {q.size()}, k:{k.size()}, v:{v.size()}")
        values, attention = scaled_dot_product(
            q, k, v, mask
        )  # values: 30 x 8 x 200 x 64
        print(f"values: {values.size()}, attention:{attention.size()}")
        values = values.reshape(
            batch_size, sequence_length, self.num_heads * self.head_dim
        )  # 30 x 200 x 512
        print(f"values after reshaping: {values.size()}")
        out = self.linear_layer(values)  # 30 x 200 x 512
        print(f"out after passing through linear layer: {out.size()}")
        return out  # 30 x 200 x 512


# TODO Encoder'dan farklı olarak multi head attention ve de cross attention gerekiyor
class MultiHeadCrossAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model, 2 * d_model)  # 1024
        self.q_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask=None):
        batch_size, sequence_length, d_model = x.size()  # 30 x 200 x 512
        print(f"x.size(): {x.size()}")
        kv = self.kv_layer(x)  # 30 x 200 x 1024
        print(f"kv.size(): {kv.size()}")
        q = self.q_layer(y)  # 30 x 200 x 512
        print(f"q.size(): {q.size()}")
        kv = kv.reshape(
            batch_size, sequence_length, self.num_heads, 2 * self.head_dim
        )  # 30 x 200 x 8 x 128
        q = q.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )  # 30 x 200 x 8 x 64
        kv = kv.permute(0, 2, 1, 3)  # 30 x 8 x 200 x 128
        q = q.permute(0, 2, 1, 3)  # 30 x 8 x 200 x 64
        k, v = kv.chunk(2, dim=-1)  # K: 30 x 8 x 200 x 64, v: 30 x 8 x 200 x 64
        values, attention = scaled_dot_product(q, k, v, mask)  #  30 x 8 x 200 x 64
        print(f"values: {values.size()}, attention:{attention.size()}")
        values = values.reshape(batch_size, sequence_length, d_model)  #  30 x 200 x 512
        out = self.linear_layer(values)  #  30 x 200 x 512
        print(f"out after passing through linear layer: {out.size()}")
        return out  #  30 x 200 x 512


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention = MultiHeadCrossAttention(
            d_model=d_model, num_heads=num_heads
        )
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob
        )
        self.norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, decoder_mask):
        _y = y  # 30 x 200 x 512
        print("MASKED SELF ATTENTION")
        y = self.self_attention(y, mask=decoder_mask)  # 30 x 200 x 512
        print("DROP OUT 1")
        y = self.dropout1(y)  # 30 x 200 x 512
        print("ADD + LAYER NORMALIZATION 1")
        y = self.norm1(y + _y)  # 30 x 200 x 512

        _y = y  # 30 x 200 x 512
        print("CROSS ATTENTION")
        y = self.encoder_decoder_attention(x, y, mask=None)  # 30 x 200 x 512
        print("DROP OUT 2")  # 30 x 200 x 512
        y = self.dropout2(y)
        print("ADD + LAYER NORMALIZATION 2")
        y = self.norm2(y + _y)  # 30 x 200 x 512

        _y = y  # 30 x 200 x 512
        print("FEED FORWARD 1")
        y = self.ffn(y)  # 30 x 200 x 512
        print("DROP OUT 3")
        y = self.dropout3(y)  # 30 x 200 x 512
        print("ADD + LAYER NORMALIZATION 3")
        y = self.norm3(y + _y)  # 30 x 200 x 512
        return y  # 30 x 200 x 512


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, mask = inputs
        for module in self._modules.values():
            y = module(x, y, mask)  # 30 x 200 x 512
        return y


class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers=1):
        super().__init__()
        self.layers = SequentialDecoder(
            *[
                DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, y, mask):
        # x : 30 x 200 x 512
        # y : 30 x 200 x 512
        # mask : 200 x 200
        y = self.layers(x, y, mask)
        return y  # 30 x 200 x 512


d_model = 512
num_heads = 8
drop_prob = 0.1
batch_size = 30
max_sequence_length = 200
ffn_hidden = 2048
num_layers = 5

x = torch.randn(
    (batch_size, max_sequence_length, d_model)
)  # English sentence positional encoded
y = torch.randn(
    (batch_size, max_sequence_length, d_model)
)  # Kannada sentence positional encoded
mask = torch.full([max_sequence_length, max_sequence_length], float("-inf"))
mask = torch.triu(mask, diagonal=1)
decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
out = decoder(x, y, mask)

print(mask)
