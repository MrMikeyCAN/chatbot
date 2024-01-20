from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from utils import tokenize, LanguageIndexMapper
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from model import LSTMModel

### * Loading and pre-processing the dataset for language detection
LD_dataset = pd.read_csv("LD.csv")

X = LD_dataset.iloc[:, 0].values
y = LD_dataset.iloc[:, 1].values

### * Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
label_mapper = LanguageIndexMapper(y_train)

### * Settings
epochs = 500
batch_size = 8
learning_rate = 0.01
hidden_size = 8
input_size = 42
output_size = 17


### Dataset is being created
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = y

    def __getitem__(self, index):
        words = tokenize(self.x_data[index])
        labels = self.y_data[index]

        return words, labels

    def __len__(self):
        return self.n_samples


# Update dataset creation
dataset = ChatDataset(X_train, y_train)

train_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMModel(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(epochs):
    for words, labels in train_loader:  # Use the DataLoader to handle batching
        words = pad_sequence(words, batch_first=True, padding_value=0).to(device)
        words = words.float()

        # Convert language labels to indices using the label_mapper instance
        labels = [label_mapper.label_to_index_func(label) for label in labels]
        labels = torch.tensor(labels).to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print(f"final loss: {loss.item():.4f}")

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_example_sentence": X,
    "languages": y,
}

FILE = "data.pth"
torch.save(data, FILE)

print(f"training complete. file saved to {FILE}")
