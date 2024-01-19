import pandas as pd
from nltk_utils import lemma, tokenize
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from model import NeuralNet
from utils import build_vocab, text_to_indices, ignore_words

texts = pd.read_csv("Tweets.csv")

### * Data preprocessing
X = texts.iloc[:, 1].values
y = texts.iloc[:, -1].values

## * Lemma and tokenize
X_tokenized = [tokenize(text) for text in X if isinstance(text, str)]


all_words = [
    lemma(w)
    for text_tokens in X_tokenized
    for w in text_tokens
    if w not in ignore_words
]

all_words = sorted(set(all_words))

### * Training the SENTENCES
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

### * Settings
epochs = 500
batch_size = 8
learning_rate = 0.01
hidden_size = 8
input_size = len(
    all_words
)  # Modified this line to use the total number of unique words
output_size = 1


### ! Pytorch model
class ChatDataset(Dataset):
    def __init__(self, X_indices, y_data):
        self.n_samples = len(X_indices)
        self.x_data = X_indices
        self.y_data = y_data

    def __getitem__(self, index):
        return torch.tensor(self.x_data[index]), torch.tensor(self.y_data[index])

    def __len__(self):
        return self.n_samples


vocab = build_vocab(X_tokenized)


# Convert texts to indices using the vocabulary
X_indices = [text_to_indices(text, vocab) for text in X if isinstance(text, str)]


dataset = ChatDataset(X_indices, y)
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

### ! Device Selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNet(input_size, hidden_size, output_size).to(device)

### ! Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(epochs):
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

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

FILE = "nlp.pth"
torch.save(model.state_dict(), FILE)
print(f"training complete. file saved to {FILE}")
