import torch.nn as nn


class ImprovedRNNModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_classes,
        num_layers=1,
        dropout_rate=0.0,
    ):
        super(ImprovedRNNModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate
        )

        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, cn) = self.lstm(x)
        out = hn[-1, :, :]  # Take the hidden state from the last time step
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc(out)

        return out
