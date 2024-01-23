# Gerekli kütüphaneleri içe aktarma
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
import numpy as np

# Veri setini yükleme
data = pd.read_csv('LD.csv')

# Metinleri ve etiketleri ayırma
texts = data['Text']
labels = data['Language']

# Etiketleri sayısal formata dönüştürme
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Eğitim ve test verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenize fonksiyonu
def tokenize(text):
    return text.split()

# Kelime dağarcığını oluşturma
vocab = build_vocab_from_iterator(map(tokenize, X_train), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# Metinleri indeks dizisine dönüştürme
def text_pipeline(x):
    return vocab(tokenize(x))

# PyTorch Dataset sınıfı
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = [torch.tensor(text_pipeline(text)) for text in texts]
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Dataset ve DataLoader oluşturma
train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: pad_sequence(x, batch_first=True))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: pad_sequence(x, batch_first=True))

# LSTM Modeli
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Model parametrelerini ayarlama
vocab_size = len(vocab)
embed_dim = 128
hidden_dim = 256
num_classes = len(np.unique(labels))

# Modeli oluşturma
model = LSTMModel(vocab_size, embed_dim, hidden_dim, num_classes)

# Loss Fonksiyonu ve Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Eğitim Fonksiyonu
def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        optimizer.zero_grad()
        output = model(texts)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Validasyon Fonksiyonu
def validate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for texts, labels in test_loader:
            output = model(texts)
            loss = criterion(output, labels)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# Eğitim Döngüsü
num_epochs = 100
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer)
    test_loss = validate(model, test_loader, criterion)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

# Modeli Kaydetme
torch.save(model.state_dict(), 'language_detection_model.pth')
