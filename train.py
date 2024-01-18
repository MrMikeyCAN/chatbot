import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, lemma
from model import NeuralNet
import torch.optim as optim
import torch.nn.functional as F

# Load intents from intents.json
with open("intents.json", "r") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Loop through each sentence in intents patterns
for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Process words
ignore_words = ["?", ".", "!"]
all_words = [lemma(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create training data
X_train = []
y_train = []
for pattern_sentence, tag in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyperparameters
num_epochs = 300
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)


# Define the chat dataset class
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.x_data = torch.from_numpy(X).float()
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# Create dataset and dataloader
dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

# Initialize device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print(f"Final loss: {loss.item():.4f}")

# Save the trained model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
}
FILE = "data.pth"
torch.save(data, FILE)
print(f"Training complete. Model saved to {FILE}")

# Your new code for text generation using PyTorch
seq = [["cloudy"]]  # Replace with your own sequence
X_gen = []
y_gen = []
total_words_dropped = 0

for i in seq:
    if len(i) > 1:
        for index in range(1, len(i)):
            X_gen.append(i[:index])
            y_gen.append(i[index])
    else:
        total_words_dropped += 1

print("Total Single Words Dropped are:", total_words_dropped)

if X_gen:
    X_gen = torch.tensor(X_gen, dtype=torch.long)
    y_gen = torch.tensor(y_gen, dtype=torch.long)

    vocab_size = len(all_words) + 1

    class TextGenerationModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_size):
            super(TextGenerationModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
            self.fc1 = nn.Linear(hidden_size, 100)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(100, vocab_size)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            lstm_out = lstm_out[:, -1, :]
            fc1_out = self.relu(self.fc1(lstm_out))
            fc2_out = self.fc2(fc1_out)
            output = self.softmax(fc2_out)
            return output

    embedding_dim_gen = 14
    hidden_size_gen = 100
    output_size_gen = len(all_words) + 1

    text_gen_model = TextGenerationModel(
        output_size_gen, embedding_dim_gen, hidden_size_gen
    )

    criterion_gen = nn.CrossEntropyLoss()
    optimizer_gen = torch.optim.Adam(text_gen_model.parameters(), lr=0.004)

    # Training loop for text generation
    for epoch in range(150):
        optimizer_gen.zero_grad()
        output_gen = text_gen_model(X_gen)
        loss_gen = criterion_gen(output_gen, y_gen)
        loss_gen.backward()
        optimizer_gen.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/150], Loss: {loss_gen.item():.4f}")

    # Save the trained PyTorch text generation model
    torch.save(text_gen_model.state_dict(), "nwp_pytorch.pth")
    print("Text generation model trained and saved.")

# Continue with any additional code as needed
