import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd

# Veri Yükleme ve Ön İşleme
file_path = 'LD.csv'  # Dosya yolunuzu buraya girin
data = pd.read_csv(file_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('complete_transformer_model.pth')
model.to(device)
model.eval()



def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(yield_tokens(data['Text']), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

def predict(text, model, tokenizer, vocab):
    # Text'i tokenize et ve tensora dönüştür
    tokens = tokenizer(text)
    indices = [vocab.stoi[token] for token in tokens]
    indices_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(1).to(device)

    # Model ile tahmin yap
    with torch.no_grad():
        output = model(indices_tensor)

    # Tahmin sonucunu al (örneğin, en olası kelime indeksi)
    predicted_index = output.argmax(1)
    predicted_word = vocab.itos[predicted_index]

    return predicted_word

# Tahmin yapılacak metni girin
input_text = input("Metin gir :")
predicted_word = predict(input_text, model, tokenizer, vocab)
print("Predicted word:", predicted_word)