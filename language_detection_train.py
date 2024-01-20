import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
import pandas as pd
from utils import tokenize, LanguageIndexMapper, text_to_speech
from model import ImprovedLSTMModel

# Loading and pre-processing the dataset for language detection
LD_dataset = pd.read_csv("LD.csv")

# Check for GPU availability and move data to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = LD_dataset["Text"].values
y = LD_dataset["Language"].values

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
label_mapper = LanguageIndexMapper(y_train)

# Settings
epochs = 500
batch_size = 8
input_size = 42
output_size = 17
num_classes = 17
learning_rate = 0.0001  # Try a smaller learning rate
hidden_size = 64  # Increased hidden size
num_layers = 3  # Increased number of layers
dropout_rate = 0.5  # Increased dropout rate

# Tokenize all data
X_train_tokenized = [tokenize(text) for text in X_train]
X_test_tokenized = [tokenize(text) for text in X_test]


# Dataset creation
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = y

    def __getitem__(self, index):
        words = self.x_data[index]
        labels = self.y_data[index]
        return words, labels

    def __len__(self):
        return self.n_samples


# Update dataset creation
train_dataset = ChatDataset(X_train_tokenized, y_train)
test_dataset = ChatDataset(X_test_tokenized, y_test)


# DataLoader with pin_memory
def collate_batch(batch):
    words, labels = zip(*batch)
    words = pad_sequence(words, batch_first=True, padding_value=0).float().to(device)
    labels = torch.tensor(
        [label_mapper.label_to_index_func(label) for label in labels], device=device
    )
    return words, labels


# DataLoader with pin_memory and parallel data loading
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


model = ImprovedLSTMModel(
    input_size, hidden_size, output_size, num_classes, num_layers, dropout_rate
).to(device)
model.apply(
    lambda x: nn.init.kaiming_uniform_(x.weight) if isinstance(x, nn.Linear) else None
)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=1e-5
)  # L2 regularization
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


# Custom Early Stopping
class EarlyStopping:
    def __init__(self, patience=20):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_model_state = None  # Store the best model state

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()  # Update best model state
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# Train the model with early stopping using custom EarlyStopping class
best_validation_loss = float("inf")
early_stopping = EarlyStopping(patience=20)

for epoch in range(epochs):
    for i, (words, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

    # Validation loss
    with torch.no_grad():
        val_losses = []
        for val_words, val_labels in test_loader:
            val_outputs = model(val_words)
            val_loss = criterion(val_outputs, val_labels)
            val_losses.append(val_loss.item())

    val_loss = sum(val_losses) / len(val_losses)

    scheduler.step(val_loss)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}"
        )

    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping. No improvement in validation loss.")
        break

text_to_speech(bot_name="Akaxtron", text=f"Final loss: {loss.item():.4f}")

# Save the best model state dict
FILE = "best_model_lstm.pth"
torch.save(early_stopping.best_model_state, FILE)

print(f"Training complete. Best model saved to {FILE}")
