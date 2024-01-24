from transformers import (
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
data = pd.read_csv("ED.csv")
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
    num_train_epochs=0.5,
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

# Modeli kaydetme
model.save_pretrained(
    "./emotion_detection_model",
    model_name="pytorch_model.bin",
    tokenizer_name="tokenizer",
)
tokenizer.save_pretrained(
    "./emotion_detection_model",
    model_name="pytorch_model.bin",
    tokenizer_name="tokenizer",
)