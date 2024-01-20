import torch.nn as nn
import torch



class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_classes, num_layers=1, dropout_rate=0.0):
        super(ImprovedLSTMModel, self).__init__()

        # Use bidirectional LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)

        # Batch Normalization for stability and faster convergence
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Forward propagate LSTM
        out, _ = self.lstm(x)

        # Take the output from the last time step
        out = out[:, -1, :]

        # Batch Normalization
        out = self.batch_norm(out)

        # Dropout for regularization
        out = self.dropout(out)

        # Fully connected layer (no ReLU activation)
        out = self.fc(out)

        return out
