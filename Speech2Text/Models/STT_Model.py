import torch
import torch.nn as nn
import torch.nn.functional as f

# Device to Use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()

        # Functions
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True,
                            device=device)

    def forward(self, x):
        # Processing
        output = self.dropout(x)
        output, hidden = self.lstm(output)
        return output, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # Functions
        self.EO = nn.Linear(hidden_size, hidden_size, device=device)
        self.Hi = nn.Linear(hidden_size * 2, hidden_size, device=device)
        self.Va = nn.Linear(hidden_size, 1, device=device)

    def forward(self, keys, query):
        # Parameter Editing
        hidden_state = query[0].permute(1, 0, 2)
        cell_state = query[1].permute(1, 0, 2)

        # Combining
        hidden_state = torch.cat((hidden_state, cell_state), dim=-1)

        # Context Calculation
        scores = self.Va(torch.tanh(self.Hi(hidden_state) + self.EO(keys)))

        weights = f.softmax(scores, dim=1)
        context = torch.sum(torch.mul(weights, keys), dim=1).unsqueeze(1)

        return context


class Decoder(nn.Module):
    def __init__(self, hidden_size, num_classes, num_layers, dropout):
        super(Decoder, self).__init__()

        # Functions
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(num_classes, hidden_size)
        self.attention = Attention(hidden_size).to(device)
        self.lstm = nn.LSTM(input_size=hidden_size*2,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True,
                            device=device)
        self.fc = nn.Linear(hidden_size, num_classes, device=device)

    def forward(self, decoder_input, decoder_hidden, encoder_outputs):
        # Processing
        decoder_input = self.dropout(self.embedding(decoder_input))

        # Attention Mechanism
        context = self.attention(encoder_outputs, decoder_hidden)
        input_lstm = torch.cat((decoder_input, context), dim=-1)

        # Decoder Processing
        decoder_output, hidden = self.lstm(input_lstm, decoder_hidden)
        decoder_output = self.out(decoder_output)

        return decoder_output, hidden


class STT(nn.Module):
    def __init__(self, input_size,
                 hidden_size, num_classes,
                 num_layers=1, dropout=0.2,
                 max_length=100, sos_token=0,
                 eos_token=1):

        super(STT, self).__init__()

        # Parameters
        self.sos = sos_token
        self.eos = eos_token

        self.max_length = max_length

        # Functions
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout).to(device)
        self.decoder = Decoder(hidden_size, num_classes, num_layers, dropout).to(device)

    def forward(self, x, target=None):
        # Encoder Calculation
        encoder_outputs, encoder_hidden = self.encoder(x)

        # Decoder Preparation
        batch_size = x.size(0)

        decoder_outputs = []

        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(self.sos)
        decoder_hidden = encoder_hidden

        # Setting max_length
        max_length = self.max_length if target is None else target.size(1)

        # Processing
        for i in range(max_length):
            # Decoder Calculation
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)

            # Setting New Input
            if target is not None:
                decoder_input = target[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(1).detach()

                if decoder_input == self.eos:
                    break

        # Output Concatenation and log_softmax Application
        decoder_outputs = torch.cat(decoder_outputs, 1)
        decoder_outputs = f.log_softmax(decoder_outputs, dim=1)
        return decoder_outputs


"""
input_size : Size of the selected audio feature
hidden_size: Number of neurons in the hidden layer
num_classes : Number of output = Alphabet length + number of special characters
num_layers : Number of hidden layers
dropout : Dropout amount
max_length : the longest number of letters it can make
sos_token : value of the starting character
eos_token : value of end character

NOTE: Since the STT model activation function is log_softmax, the loss function should be NLLLoss
"""