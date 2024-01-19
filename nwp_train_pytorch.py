import torch
import torch.nn as nn
import torch.optim as optim
import torchtext

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the text data
with open("text.txt", "r", encoding="utf-8") as myfile:
    mytext = myfile.read()

# Tokenization using PyTorch
mytokenizer = torchtext.data.utils.get_tokenizer("basic_english")
tokenized_text = mytokenizer(mytext)
vocab = set(tokenized_text)

word_to_index = {word: idx for idx, word in enumerate(vocab)}
index_to_word = {idx: word for idx, word in enumerate(vocab)}

# Convert text to numerical sequences
numerical_text = [word_to_index[word] for word in tokenized_text]

# Define max_sequence_len
max_sequence_len = 10  # You can replace this with the appropriate value

# Prepare input sequences
input_sequences = []
for i in range(1, len(numerical_text)):
    sequence = numerical_text[i - max_sequence_len : i]
    input_sequences.append(sequence)

# Filter out sequences that don't have enough elements
input_sequences = [seq for seq in input_sequences if len(seq) == max_sequence_len]

# Convert input sequences to PyTorch tensors and move to GPU
X = torch.LongTensor([sequence[:-1] for sequence in input_sequences]).to(device)
y = torch.LongTensor([sequence[-1] for sequence in input_sequences]).to(device)

# Define the dimensions
embedding_dim = 100
hidden_dim = 150


# Define the RNN model using PyTorch and move to GPU
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        output = self.fc(rnn_out[:, -1, :])
        return output


# Instantiate the model and move to GPU
model = RNNModel(len(vocab), embedding_dim, hidden_dim).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()


# Function to generate next words
def nwp(input_text, predict_next_words):
    tokenized_input = mytokenizer(input_text)
    for _ in range(predict_next_words):
        input_sequence = [word_to_index.get(word, 0) for word in tokenized_input]
        input_sequence = torch.LongTensor(input_sequence[-max_sequence_len + 1 :]).to(
            device
        )

        output = model(input_sequence.unsqueeze(0))
        predicted_index = torch.argmax(output).item()
        output_word = index_to_word.get(predicted_index, "<unknown>")

        input_text += " " + output_word
        tokenized_input.append(output_word)

    return input_text


# Move the model back to CPU for generating text
model = model.to("cpu")

# Example usage
generated_text = nwp("Hello", 5)
print(generated_text)
