import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, lemma
from model import NeuralNet
import tensorflow as tf

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
num_epochs = 800
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

# Your new code for text generation
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

X_gen = tf.keras.preprocessing.sequence.pad_sequences(X_gen)
y_gen = tf.keras.utils.to_categorical(y_gen)

vocab_size = len(all_words) + 1

model_lstm = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(vocab_size, 14),
        tf.keras.layers.LSTM(100, return_sequences=True),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(vocab_size, activation="softmax"),
    ]
)
model_lstm.summary()

model_lstm.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.004),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model_lstm.fit(X_gen, y_gen, epochs=150)

model_lstm.save("nwp.h5")
vocab_array = np.array(all_words)

# Continue with any additional code as needed
