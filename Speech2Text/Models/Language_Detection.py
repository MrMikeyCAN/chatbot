import torch
import torch.nn as nn

# Device to Use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Non-model Parameters
softmax = nn.Softmax(dim=1)


class Language(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout=0.2):
        super(Language, self).__init__()

        # Parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Functions
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True,
                            device=device)
        self.fc = nn.Linear(hidden_size, num_classes, device=device)

    def forward(self, x):
        # Initial Parameters
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Processing
        x = self.dropout(x)
        x, _ = self.lstm(x, (h0, c0))
        x = self.fc(x[:, -1, :])
        return x


"""
input_size : Size of the selected audio feature
hidden_size: Number of neurons in the hidden layer
num_classes : Number of outputs (Number of languages)
num_layers : Number of hidden layers
dropout : Dropout amount

NOTE: After this model is printed, it is necessary to use the softmax function in the testing process.
"""