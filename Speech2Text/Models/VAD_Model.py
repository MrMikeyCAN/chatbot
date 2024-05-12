import torch
import torch.nn as nn

# Kullanılacak olan cihaz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model dışı parametreler
sigmoid = nn.Sigmoid()
threshold = 0.7


class VAD(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=1, num_layers=1, dropout=0.2):
        super(VAD, self).__init__()

        # Parametreler
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Fonksiyonlar
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True,
                            device=device)
        self.fc = nn.Linear(hidden_size, num_classes, device=device)

    def forward(self, x):
        # Başlangıç parametreleri
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # İşlemler
        x = self.dropout(x)
        x, _ = self.lstm(x, (h0, c0))
        x = self.fc(x[:, -1, :])
        return x


"""
input_size : Seçilen ses özelliğinin boyutu
hidden_size : Gizli katmandaki nöron sayısı
num_classes : Çıktı sayısı = 1
num_layers : Gizli katman sayısı
dropout : Dropout miktarı

NOT : Bu model çıktı alındıktan sonra test etme işleminde sigmodi fonksiyonu kullanılması gerekli
"""
