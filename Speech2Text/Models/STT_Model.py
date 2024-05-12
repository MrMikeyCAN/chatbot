import torch
import torch.nn as nn
import torch.nn.functional as f

# Kullanılacak olan cihaz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()

        # Fonksiyonlar
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True,
                            device=device)

    def forward(self, x):
        # İşlemler
        output = self.dropout(x)
        output, hidden = self.lstm(output)
        return output, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # Fonksiyonlar
        self.EO = nn.Linear(hidden_size, hidden_size, device=device)
        self.Hi = nn.Linear(hidden_size * 2, hidden_size, device=device)
        self.Va = nn.Linear(hidden_size, 1, device=device)

    def forward(self, keys, query):
        # Parametre düzenlemesi
        hidden_state = query[0].permute(1, 0, 2)
        cell_state = query[1].permute(1, 0, 2)

        # Birleştirme
        hidden_state = torch.cat((hidden_state, cell_state), dim=-1)

        # Context hesaplaması
        scores = self.Va(torch.tanh(self.Hi(hidden_state) + self.EO(keys)))

        weights = f.softmax(scores, dim=1)
        context = torch.sum(torch.mul(weights, keys), dim=1).unsqueeze(1)

        return context


class Decoder(nn.Module):
    def __init__(self, hidden_size, num_classes, num_layers, dropout):
        super(Decoder, self).__init__()

        # Fonksiyonlar
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(num_classes, hidden_size)
        self.attention = Attention(hidden_size).to(device)
        self.lstm = nn.LSTM(input_size=hidden_size*2,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True,
                            device=device)
        self.fc = nn.Linear(hidden_size, num_classes, device=device)

    def forward(self, decoder_input, decoder_hidden, encoder_outputs):
        # İşleme
        decoder_input = self.dropout(self.embedding(decoder_input))

        # Attention mekanizması
        context = self.attention(encoder_outputs, decoder_hidden)
        input_lstm = torch.cat((decoder_input, context), dim=-1)

        # Decoder işlemi
        decoder_output, hidden = self.lstm(input_lstm, decoder_hidden)
        decoder_output = self.out(decoder_output)

        return decoder_output, hidden


class STT(nn.Module):
    def __init__(self, input_size,
                 hidden_size, num_classes,
                 num_layers=1, dropout=0.2,
                 max_length=100, sos_token=0,
                 eos_token=1):

        super(STT, self).__init__()

        # Parametreler
        self.sos = sos_token
        self.eos = eos_token

        self.max_length = max_length

        # Fonksiyonlar
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout).to(device)
        self.decoder = Decoder(hidden_size, num_classes, num_layers, dropout).to(device)

    def forward(self, x, target=None):
        # Encoder hesaplama
        encoder_outputs, encoder_hidden = self.encoder(x)

        # Decoder hazırlık
        batch_size = x.size(0)

        decoder_outputs = []

        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(self.sos)
        decoder_hidden = encoder_hidden

        # max_length ayarlama
        max_length = self.max_length if target is None else target.size(1)

        # İşlemeler
        for i in range(max_length):
            # Decoder hesaplama
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)

            # Yeni input ayaralama
            if target is not None:
                decoder_input = target[:, i].unsqueeze(1)
            else:
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(1).detach()

                if decoder_input == self.eos:
                    break

        # Output birleştirme ve log_softmax uygulama
        decoder_outputs = torch.cat(decoder_outputs, 1)
        decoder_outputs = f.log_softmax(decoder_outputs, dim=1)
        return decoder_outputs


"""
input_size : Seçilen ses özelliğinin boyutu
hidden_size : Gizli katmandaki nöron sayısı
num_classes : Çıktı sayısı = Alfabe uzunluğu + özel karakter sayısı
num_layers : Gizli katman sayısı
dropout : Dropout miktarı
max_length : yapabileceği en uzun harf sayısı
sos_token : başlangıç karakterinin değeri
eos_token : bitiş karakterinin değeri
"""