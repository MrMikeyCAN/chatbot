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