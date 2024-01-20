import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()

        # Use an LSTM layer instead of linear layers
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Initialize hidden state with zeros
        # Make sure h0 and c0 are 2-D tensors
        h0 = torch.zeros(x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(x.size(0), self.lstm.hidden_size).to(x.device)

        # Expand dimensions for batch_first=True
        h0 = h0.unsqueeze(0)
        c0 = c0.unsqueeze(0)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Take the output from the last time step
        out = out[:, -1, :]

        # Fully connected layer
        out = self.fc(out)

        # ReLU activation
        out = self.relu(out)

        return out
