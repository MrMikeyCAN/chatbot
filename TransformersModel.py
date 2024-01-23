import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
import time
import os

# Veri Yükleme ve Ön İşleme
file_path = 'LD.csv'  # Dosya yolunuzu buraya girin
data = pd.read_csv(file_path)

# Tokenizer ve Vocab Oluşturma
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(data['Text']), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# Veriyi Tensor Formatına Dönüştürme
def data_process(texts):
    return torch.cat([torch.tensor(vocab(tokenizer(text)), dtype=torch.long)
                      for text in texts])

train_data = data_process(data['Text'])

# Batchify Fonksiyonu
def batchify(data, bsz):
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data

batch_size = 20
train_data = batchify(train_data, batch_size)

# Positional Encoding Class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Transformer Model Class
class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output

# Model ve Eğitim Parametreleri
ntokens = len(vocab)  # vocabulary boyutu
emsize = 200  # embedding boyutu
d_hid = 200  # nn.TransformerEncoder içindeki feedforward network boyutu
nlayers = 2  # nn.TransformerEncoder içindeki nn.TransformerEncoderLayer sayısı
nhead = 2  # nn.MultiheadAttention içindeki başlık (head) sayısı
dropout = 0.2  # dropout olasılığı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

# Eğitim Fonksiyonları ve Parametrelerin Güncellenmesi
def train(model: nn.Module, train_data: Tensor, bptt: int, criterion, optimizer):
    model.train()  # Eğitim moduna geçiş
    total_loss = 0.
    start_time = time.time()
    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}')
            total_loss = 0
            start_time = time.time()


def get_batch(source: Tensor, i: int, bptt: int) -> tuple[Tensor, Tensor]:
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data.to(device), target.to(device)

# Eğitim Döngüsü ve Optimizer'ın Güncellenmesi
bptt = 35
criterion = nn.CrossEntropyLoss()
lr = 0.001  # Öğrenme oranı (düşük tutulabilir)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 30  # toplam eğitim epoch sayısı (gerektiği kadar ayarlayın)
log_interval = 200  # Loglama aralığı
for epoch in range(1, epochs + 1):
    train(model, train_data, bptt, criterion, optimizer)