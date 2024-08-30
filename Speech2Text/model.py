import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset
import audio
from torchsummary import summary


class ConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size,
                               kernel_size=3, padding=1, bias=False,
                               device=dataset.device, dtype=dataset.x_dtype)
        self.bn1 = nn.BatchNorm1d(hidden_size, device=dataset.device, dtype=dataset.x_dtype)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size,
                               kernel_size=3, stride=2, padding=1, bias=False,
                               device=dataset.device, dtype=dataset.x_dtype)

        self.bn2 = nn.BatchNorm1d(hidden_size, device=dataset.device, dtype=dataset.x_dtype)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)
        return x


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, layer_size, output_size, p):
        super(Model, self).__init__()

        self.conv = ConvEncoder(input_size, hidden_size).to(dataset.device, dtype=dataset.x_dtype)
        self.lstm = nn.LSTM(hidden_size, hidden_size,
                            layer_size, batch_first=True, dropout=p if layer_size > 1 else 0,
                            device=dataset.device, dtype=dataset.x_dtype)
        self.ln = nn.LayerNorm(hidden_size, device=dataset.device, dtype=dataset.x_dtype)
        self.dropout = nn.Dropout(p)
        self.fc = nn.Linear(hidden_size, output_size,
                            device=dataset.device, dtype=dataset.x_dtype)

    def forward(self, x):
        output = self.conv(x)
        output, hidden = self.lstm(output)
        output = self.ln(output)
        output = self.dropout(output)
        output = self.fc(output)
        return output


input_size = audio.N_MELS
hidden_size = 128
layer_size = 1
output_size = len(dataset.train_dataset.alphabet)+1
p = 0.2

model = Model(input_size, hidden_size, layer_size, output_size, p).to(dataset.device)

total_params = sum(p.numel() for p in model.parameters())
print(total_params)

for x, y in dataset.train_dataloader:
    start = time.time()
    predict = model(x)
    print(predict.shape)
    end = time.time()
    print((end - start))
