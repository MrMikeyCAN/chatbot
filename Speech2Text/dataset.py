import torch
import pandas as pd
import audio
import tokinizer
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_dtype = torch.float16
y_dtype = torch.int8
batch_size = 1
shuffle = True


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file: str,
                 path: str,
                 alphabet_file: str,
                 x_feature: str,
                 y_feature: str,
                 frac: float = 1,
                 sep: str = None,
                 x_dtype: torch.dtype = torch.float16,
                 y_dtype: torch.dtype = torch.int8):

        assert 0 <= frac <= 1, 'frac must be between 0 and 1'

        self.data_path = path

        df = pd.read_csv(file, sep=sep)
        df = df.sample(frac=frac).reset_index(drop=True)

        self.x_feature = df[x_feature]
        self.y_feature = df[y_feature]

        self.alphabet = tokinizer.get_alphabet(alphabet_file)

        self.x_dtype = x_dtype
        self.y_dtype = y_dtype

    def __len__(self):
        return len(self.x_feature)

    def __getitem__(self, idx):
        waveform = audio.load_audio(os.path.join(self.data_path,
                                                 self.x_feature[idx]))
        log_mel_spectrogram = audio.log_mel_spectrogram(waveform, audio.mel_spectrogram, dtype=self.x_dtype)

        tokenized_tensor = tokinizer.tokenize(self.y_feature[idx], self.alphabet, dtype=self.y_dtype)

        return log_mel_spectrogram, tokenized_tensor


def collect_fn(batch):
    x = torch.nn.utils.rnn.pad_sequence([x[0] for x in batch], batch_first=True)
    y = torch.nn.utils.rnn.pad_sequence([y[1] for y in batch], batch_first=True)

    return x.to(device), y.to(device)


train_dataset = Dataset(file='cv-corpus-18.0-2024-06-14/tr/train.tsv',
                        path='cv-corpus-18.0-2024-06-14/tr/clips',
                        alphabet_file='alphabet.txt',
                        x_feature='path',
                        y_feature='sentence',
                        frac=0.1,
                        sep='\t')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=shuffle, collate_fn=collect_fn)