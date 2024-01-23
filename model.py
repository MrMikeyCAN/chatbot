<<<<<<< HEAD
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, embedding_dim, vocab_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Embedding
        x = self.embedding(x)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Get output from the last time step
        out = self.fc(out[:, -1, :])
        return out
=======
import torch.nn as nn


class ImprovedTransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_classes, num_layers=1, dropout_rate=0.0):
        super(ImprovedTransformerModel, self).__init__()

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=1, batch_first=True
        )

        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(input_size)

        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.transformer_layer(x)
        out = out[:, -1, :]
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc(out)

        return out
>>>>>>> 1f692eadf1f938ce28fe03ddd7b6407792552e62
