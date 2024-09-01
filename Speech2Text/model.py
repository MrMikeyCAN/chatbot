import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset
import audio
from torchsummary import summary


class ConvEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size,
                               kernel_size=3, padding=1, bias=False,
                               device=dataset.device, dtype=dataset.x_dtype)
        self.bn1 = nn.BatchNorm1d(hidden_size, device=dataset.device, dtype=dataset.x_dtype)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size,
                               kernel_size=3, padding=1, bias=False,
                               device=dataset.device, dtype=dataset.x_dtype)

        self.bn2 = nn.BatchNorm1d(hidden_size, device=dataset.device, dtype=dataset.x_dtype)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)
        return x


class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, p: float):
        super(Model, self).__init__()

        self.conv = ConvEncoder(input_size, hidden_size).to(dataset.device, dtype=dataset.x_dtype)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True,
                            device=dataset.device, dtype=dataset.x_dtype)
        self.ln = nn.LayerNorm(hidden_size, device=dataset.device, dtype=dataset.x_dtype)
        self.dropout = nn.Dropout(p)
        self.fc = nn.Linear(hidden_size, output_size,
                            device=dataset.device, dtype=dataset.x_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv(x)
        output, hidden = self.lstm(output)
        output = self.ln(output)
        output = self.dropout(output)
        output = self.fc(output)
        return output
