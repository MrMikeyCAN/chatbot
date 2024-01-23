import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd

# Veri setinizi yükleyin ve ön işleme yapın
# Bu adımda metin verilerinizi ve etiketlerinizi hazırlamalısınız
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

# Veriyi tensorlara dönüştürme (örnek olarak kelime seviyesi gömme kullanımı)
embedding_dim = 128
hidden_dim = 256
num_classes = 17

class LanguageDetectionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(LanguageDetectionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Sadece son zaman adımının çıktısını kullanın
        out = self.fc(out)
        return out

# Modeli oluşturun
model = LanguageDetectionModel(len(vocab), embedding_dim, hidden_dim, num_classes)

# Kayıp fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim döngüsü
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:  # Eğitim verilerini kullanarak modeli güncelleyin
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Modelin test edilmesi
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:  # Test verilerini kullanarak modelin performansını değerlendirin
        inputs, labels = batch
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test verileri üzerinde doğruluk: {accuracy * 100:.2f}%')
