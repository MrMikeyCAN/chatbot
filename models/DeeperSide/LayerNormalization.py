import torch
from torch import nn


# ! Layer normalization ile ulaşmaya çalıştığımız veriler aslında bizim outputta çıkmasını öngördüğümüz ve eğitmek için eğitim verilerinin bulunduğu katmandır.Bu kısım bolca matematik içerdiğinden pek de bir açıklama yapacağım konu bulunmuyor

# TODO örnek bir layer norm oluşturuyoruz ama temelde yapacağımız bunu büyük verilere adapte etmek olacak

inputs = torch.Tensor([[[0.2, 0.1, 0.3], [0.5, 0.1, 0.1]]])
B, S, E = inputs.size()
inputs = inputs.reshape(S, B, E)
# print(inputs.size())

# * öğrenebilir verilerimiz bunlar olacak

parameter_shape = inputs.size()[-2:]
gamma = nn.Parameter(torch.ones(parameter_shape))
beta = nn.Parameter(torch.zeros(parameter_shape))


# print(gamma.size(), beta.size())

# TODO dimmensionlar alacağız
dims = [-(i + 1) for i in range(len(parameter_shape))]

# TODO ortalamaları alacağız
mean = inputs.mean(dim=dims, keepdim=True)
# print(mean.size())

# print(mean)

# TODO STD değerini bulacağız

var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
epsilon = 1e-5
std = (var + epsilon).sqrt()
# print(std)

# TODO Gama ve Y değerlerini bu işlemlere tabii tutarak out fonksiyonuna hazırlıyoruz
y = (inputs - mean) / std
# print(y)

out = gamma * y + beta
# print(out)

# TODO şimdide yukarıdaki örneği burada class haline getiriyoruz


class LayerNormalization:
    def __init__(self, parameters_shape, eps=1e-5):
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, input):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        print(f"Mean \n ({mean.size()}): \n {mean}")
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        print(f"Standard Deviation \n ({std.size()}): \n {std}")
        y = (inputs - mean) / std
        print(f"y \n ({y.size()}) = \n {y}")
        out = self.gamma * y + self.beta
        print(f"out \n ({out.size()}) = \n {out}")
        return out


batch_size = 3
sentence_length = 5
embedding_dim = 8
inputs = torch.randn(sentence_length, batch_size, embedding_dim)

print(f"input \n ({inputs.size()}) = \n {inputs}")
# ! İstersek -1 için layer_norm istersek de -2 için uygulayabiliriz ikisi de farklı sonuçlar çıkardığından yeni bir seçenek doğar
layer_norm = LayerNormalization(inputs.size()[-1:])

out = layer_norm.forward(inputs)

print(out[0].mean(), out[0].std())
