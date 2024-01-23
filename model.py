import torch
import torch.nn as nn

class ImprovedTransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_classes, num_layers=1, dropout_rate=0.0):
        super(ImprovedTransformerModel, self).__init__()

        # Set batch_first=True in the transformer layer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=1, batch_first=True  # <-- Set batch_first=True
        )

        # Batch Normalization for stability and faster convergence
        self.batch_norm = nn.BatchNorm1d(input_size)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        # Forward propagate transformer layer
        out = self.transformer_layer(x)

        # Take the output from the last time step
        out = out[:, -1, :]

        # Batch Normalization
        out = self.batch_norm(out)

        # Dropout for regularization
        out = self.dropout(out)

        # Fully connected layer (no ReLU activation)
        out = self.fc(out)

        return out
