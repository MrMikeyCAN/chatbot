"""from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Veri yükleme ve hazırlama
data = pd.read_csv("LD.csv")
label_encoder = LabelEncoder()
data["encoded_labels"] = label_encoder.fit_transform(data["labels"])
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Tokenizer ve model yükleme
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased", num_labels=len(label_encoder.classes_)
)


# Veriyi tokenleştirme
class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = CustomDataset(
    train_data["target"].tolist(), train_data["encoded_labels"].tolist()
)
test_dataset = CustomDataset(
    test_data["target"].tolist(), test_data["encoded_labels"].tolist()
)

# Eğitim ayarları
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Eğitim
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# Save label encoder
torch.save(
    label_encoder.classes_, "./language_detection_model/label_encoder_classes.pt"
)

# Modeli kaydetme
model.save_pretrained("./language_detection_model")
tokenizer.save_pretrained("./language_detection_model")
"""

import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
class LanguageClassifier(nn.Module):
    def __init__(self,input,hidden,output):
        super(LanguageClassifier, self).__init__()
        self.linear1 = nn.Linear(input,hidden)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden,hidden)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden,output)
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x
    
# Create ANN
data = pd.read_csv("LD.csv")
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
cross_validation,test_data = train_test_split(test_data, test_size=0.5, random_state=42)
train_data = train_data.to_numpy()
cross_validation = cross_validation.to_numpy()
test_data = test_data.to_numpy()



# Tokenize edilmiş metni tekrar birleştir
metin = "Merhaba, nasılsınız? Umarım gününüz iyi geçiyordur."

# CountVectorizer nesnesini oluştur
vectorizer = CountVectorizer()

# Metni sayısal vektöre dönüştür
X = vectorizer.fit_transform([str(train_data[:,0])])

# Dönüştürülmüş vektörü görüntüle
print(X.toarray())

# Kelime dağarcığını görüntüle
print(vectorizer.get_feature_names_out())
