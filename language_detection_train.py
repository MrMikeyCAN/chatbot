import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
from model import ImprovedRNNModel
from utils import LanguageIndexMapper

# Settings
batch_size = 8
epochs = 5  # Increased the number of epochs
learning_rate = 5e-4  # Adjusted the learning rate
hidden_size = 64
num_layers = 2  # Adjusted the number of layers
dropout_rate = 0.3
max_length = 512  # Assuming a maximum sequence length

# Loading and pre-processing the dataset for language detection
LD_dataset = pd.read_csv("LD.csv")

# Check for GPU availability and move data to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
le = LabelEncoder()

X = LD_dataset["Text"].values
y = le.fit_transform(LD_dataset["Language"])

# Create an instance of LanguageIndexMapper
label_mapper = LanguageIndexMapper(y)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tokenize all data using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
X_train_tokenized = [
    tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    for text in X_train
]
X_test_tokenized = [
    tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    for text in X_test
]


# Dataset creation
class ChatDataset(Dataset):
    def __init__(self, X, y, label_mapper):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = y
        self.label_mapper = label_mapper

    def __getitem__(self, index):
        tokens = self.x_data[index]
        labels = self.y_data[index]
        return tokens, labels

    def __len__(self):
        return self.n_samples


# Update dataset creation
train_dataset = ChatDataset(X_train_tokenized, y_train, label_mapper)
test_dataset = ChatDataset(X_test_tokenized, y_test, label_mapper)


# DataLoader with pin_memory and parallel data loading
def collate_batch(batch):
    tokens, labels = zip(*batch)

    # Pad the sequences and convert to tensor
    tokens = {
        key: pad_sequence(
            [t[key].view(-1) for t in tokens], batch_first=True, padding_value=0
        ).to(device)
        for key in tokens[0].keys()
    }
    labels = torch.tensor(
        [label_mapper.label_to_index_func(label) for label in labels], device=device
    )

    return tokens, labels


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=torch.cuda.device_count(),
    collate_fn=collate_batch,
    pin_memory=True,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=torch.cuda.device_count(),
    collate_fn=collate_batch,
    pin_memory=True,
)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased", num_labels=len(le.classes_)
).to(device)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# Loss function
criterion = nn.CrossEntropyLoss()

# Train the BERT model
best_validation_loss = float("inf")
patience = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        input_ids = batch[0]["input_ids"].to(device)
        attention_mask = batch[0]["attention_mask"].to(device)
        labels = batch[1].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1)  # Gradient clipping
        optimizer.step()

    average_loss = total_loss / len(train_loader)
    print(f"Training Loss (BERT): {average_loss}")

    # Validation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Validation"):
            input_ids = batch[0]["input_ids"].to(device)
            attention_mask = batch[0]["attention_mask"].to(device)
            labels = batch[1].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy (BERT): {accuracy}")

    scheduler.step()

    if epoch > patience and average_loss > best_validation_loss:
        print(
            f"Early stopping. No improvement in validation loss for {patience} epochs."
        )
        break

    if average_loss < best_validation_loss:
        best_validation_loss = average_loss

# Save the BERT model
model.save_pretrained("language_detection_bert")
tokenizer.save_pretrained("language_detection_bert")

# Save label encoder
joblib.dump(le, "label_encoder.pkl")


# Instantiate the RNN model
num_classes = len(le.classes_)
rnn_model = ImprovedRNNModel(
    input_size=hidden_size,
    hidden_size=hidden_size,
    output_size=num_classes,
    num_layers=num_layers,
    dropout_rate=dropout_rate,
).to(device)

# Optimizer and scheduler for RNN
optimizer_rnn = torch.optim.AdamW(rnn_model.parameters(), lr=learning_rate)
scheduler_rnn = torch.optim.lr_scheduler.StepLR(optimizer_rnn, step_size=20, gamma=0.1)

# Loss function for RNN
criterion_rnn = nn.CrossEntropyLoss()

# Train the RNN model
for epoch in range(epochs):
    rnn_model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        input_ids = batch[0]["input_ids"].to(device)
        attention_mask = batch[0]["attention_mask"].to(device)
        labels = batch[1].to(device)

        optimizer_rnn.zero_grad()
        rnn_outputs = rnn_model(input_ids)
        loss_rnn = criterion_rnn(rnn_outputs, labels)
        total_loss += loss_rnn.item()

        loss_rnn.backward()
        clip_grad_norm_(rnn_model.parameters(), max_norm=1)  # Gradient clipping
        optimizer_rnn.step()

    average_loss_rnn = total_loss / len(train_loader)
    print(f"Training Loss (RNN): {average_loss_rnn}")

    # Validation
    rnn_model.eval()
    all_preds_rnn = []
    all_labels_rnn = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Validation"):
            input_ids = batch[0]["input_ids"].to(device)
            attention_mask = batch[0]["attention_mask"].to(device)
            labels = batch[1].to(device)

            rnn_outputs = rnn_model(input_ids)
            preds_rnn = torch.argmax(rnn_outputs, dim=1)

            all_preds_rnn.extend(preds_rnn.cpu().numpy())
            all_labels_rnn.extend(labels.cpu().numpy())

    accuracy_rnn = accuracy_score(all_labels_rnn, all_preds_rnn)
    print(f"Validation Accuracy (RNN): {accuracy_rnn}")

    scheduler_rnn.step()

# Save the RNN model
torch.save(rnn_model.state_dict(), "language_detection_rnn.pth")
