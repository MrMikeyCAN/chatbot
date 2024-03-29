import torch
import torch.nn as nn

# * Maksimum kelime sayısı
max_sequence_length = 10
# ! Normalde 512 olması gerekir ama eğitim dolayısıyla 6
d_model = 6


# ! Bu fonksiyonun amacı her bir attention mekanizması için olasılıkları bulmaktır
# TODO Tek değerler için cos, çift değerler için cos kullanıcaz ve değerleri öncesinde 1000'e bölüyoruz

# TODO tek mi çift mi olduğunu öğreniyoruz
even_i = torch.arange(0, d_model, 2).float()
# print(even_i)


even_denominator = torch.pow(10000, even_i / d_model)
# print(even_denominator)


# TODO Tek değerleri buluyoruz
odd_i = torch.arange(1, d_model, 2).float()
# print(odd_i)


even_denominator = torch.pow(10000, (odd_i - 1) / d_model)
# print(even_denominator)

# ! Sonuçlar söylüyor ki çiftler de tekler de günün sonunda aynı değeri veriyor bu yüzden biz de varsayılan olarak sadece bir tanesini kullanıcaz

denominator = even_denominator

# TODO Olası pozisyonları buluyoruz

position = torch.arange(max_sequence_length, dtype=torch.float).reshape(
    max_sequence_length, 1
)

# print(position)

# TODO Pozisitional encoding için gerekli sin ve cos fonksiyonlarını oluşturuyoruz

even_PE = torch.sin(position / denominator)
odd_PE = torch.cos(position / denominator)

# * Günün sonunda tekler ile çiftler arasında bir miktar farklılıklar ve olası pozisyonlar oluştu

# print(even_PE)
# print(even_PE.shape)
# print(odd_PE)
# print(odd_PE.shape)

# TODO Tensorleri üst üste bindirip birleşitriyoruz
stacked = torch.stack([even_PE, odd_PE], dim=2)
# print(stacked.shape)


PE = torch.flatten(stacked, start_dim=1, end_dim=2)
# print(PE)

# TODO Gerekli tüm kodları yazdıktan sonra PE class'ını oluşturalım


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i / self.d_model)
        position = torch.arange(self.max_sequence_length).reshape(
            self.max_sequence_length, 1
        )
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE


pe = PositionalEncoding(d_model=6, max_sequence_length=10)
# ! Her bir parçaladığımız bölüm ve cümle için seçenekler ve olası pozisyonlar oluştu
print(pe())
