import torch
import pandas as pd
import audio
import tokinizer
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_dtype = torch.float32
y_dtype = torch.int8
batch_size = 16 # çok büyük batch size bellek aşımı yapıyor
frac = 1
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
    x_list = [x[0] for x in batch]
    y_list = [y[1] for y in batch]

    x_padded = torch.nn.utils.rnn.pad_sequence(x_list, batch_first=True)
    y_padded = torch.nn.utils.rnn.pad_sequence(y_list, batch_first=True)

    input_lengths = torch.LongTensor([len(x) for x in x_list])
    target_lengths = torch.LongTensor([len(y) for y in y_list])

    return x_padded.to(device), y_padded.to(device), input_lengths.to(device), target_lengths.to(device)


train_dataset = Dataset(file='cv-corpus-18.0-2024-06-14/tr/train.tsv',
                        path='cv-corpus-18.0-2024-06-14/tr/clips',
                        alphabet_file='alphabet.txt',
                        x_feature='path',
                        y_feature='sentence',
                        x_dtype=x_dtype,
                        y_dtype=y_dtype,
                        frac=frac,
                        sep='\t')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=shuffle, collate_fn=collect_fn)

dev_dataset = Dataset(file='cv-corpus-18.0-2024-06-14/tr/dev.tsv',
                      path='cv-corpus-18.0-2024-06-14/tr/clips',
                      alphabet_file='alphabet.txt',
                      x_feature='path',
                      y_feature='sentence',
                      frac=frac,
                      x_dtype=x_dtype,
                      y_dtype=y_dtype,
                      sep='\t')

dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size,
                                             shuffle=shuffle, collate_fn=collect_fn)

test_dataset = Dataset(file='cv-corpus-18.0-2024-06-14/tr/test.tsv',
                       path='cv-corpus-18.0-2024-06-14/tr/clips',
                       alphabet_file='alphabet.txt',
                       x_feature='path',
                       y_feature='sentence',
                       frac=frac,
                       x_dtype=x_dtype,
                       y_dtype=y_dtype,
                       sep='\t')

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=shuffle, collate_fn=collect_fn)