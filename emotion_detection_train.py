import pickle
import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from models.OurModel import (
    Transformer,
    TrainingArguments,
    TransformerModelArguments,
    Trainer,
)

# TODO Veri yükleme ve hazırlama
data = pd.read_csv("ED.csv")
X = data["target"]
y = data["labels"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=2, train_size=0.1
)

X_train = X_train.tolist()
y_train = y_train.tolist()


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
## TODO girdileri oluşturuyoruz
input_ids = tokenizer.encode(
    X_train, add_special_tokens=True, max_length=2048, truncation=True
)
target_ids = tokenizer.encode(
    y_train, add_special_tokens=True, max_length=2048, truncation=True
)

# Tensorlara dönüştürelim
input_tensor = torch.tensor([input_ids])
target_tensor = torch.tensor([target_ids])

# Eğitim için model parametrelerini belirleyelim
transformerModelArguments = TransformerModelArguments(
    embed_size=1024,
    heads=8,
    dropout=0.1,
    forward_expansion=4,
    src_vocab_size=torch.max(input_tensor)
    + 1,  # +1, çünkü 0'dan başlamak yerine 1'den başlıyoruz
    trg_vocab_size=torch.max(target_tensor)
    + 1,  # +1, çünkü 0'dan başlamak yerine 1'den başlıyoruz
    src_pad_idx=0,
    trg_pad_idx=0,
    num_layers=6,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    max_length=1024,
    norm_activite_func="layernorm",
    guessLanguage=True,
)


with open("transformer_model_args.pkl", "wb") as f:
    pickle.dump(transformerModelArguments, f)

# Modeli oluşturalım
model = Transformer(transformerModelArguments)

# Eğitim için kayıp fonksiyonunu belirleyelim
criterion = nn.CrossEntropyLoss()

# Optimizasyon fonksiyonunu belirleyelim
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Eğitim verilerini belirleyelim
input_data = input_tensor.to(transformerModelArguments.device)
target_data = target_tensor.to(transformerModelArguments.device)

trainingArguments = TrainingArguments(
    epochs=10,
    model=model,
    visualize=False,
    input_data=input_tensor,
    target_data=target_tensor,
    checkpoints=5,
)

# Modeli eğitelim
trainer = Trainer(args=trainingArguments)
trainer.train()
