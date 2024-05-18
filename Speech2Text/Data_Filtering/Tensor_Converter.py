import torch
from torch.functional import F
import torchaudio
from torch.nn.functional import one_hot
from torch.utils.data import TensorDataset, DataLoader
import warnings
import pandas as pd
import json
import os
import re

# Close Warnings
warnings.filterwarnings("ignore")


class TensorConverter:
    def __init__(self, sample_fraction: float, batch_size: int,
                 json_filename: str, process: str,
                 language: int = None, alphabet: str = None):

        # Parameters
        self.sample_fraction = sample_fraction
        self.batch_size = batch_size
        self.process = process

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reading Json File and Getting Items
        try:
            with open(json_filename, 'r', encoding='utf-8') as f:
                json_file = json.load(f)[self.process]

            self.train_only = json_file['train_only']
            self.has_val = json_file['has_val']
            self.root_dir = json_file['root_dir']
            self.filenames = json_file['filenames']
            self.labels = json_file['labels']

        except FileNotFoundError:
            raise FileNotFoundError("Json File Not Found.")
        except KeyError:
            raise KeyError("Please Control Json File.")

        if self.process == "STT":
            if language is not None:
                if language > len(self.labels):
                    self.language = 0
                elif language < len(self.labels):
                    self.language = 0
                else:
                    self.language = language
            else:
                self.language = 0

        # Dataloaders
        self.train_dataloader = []
        self.test_dataloader = []
        self.val_dataloader = []

        self.input_size = None

        # For STT Labels
        if alphabet is not None:
            self.alphabet = alphabet
        else:
            # Not Important Warning
            self.alphabet = "abcdefghijklmnopqrstuvwxyzçşıüöğ "

        self.SOS_token = 0
        self.EOS_token = 1

    def process_data(self):
        # Processing Dataloaders
        self.train_dataloader = self.__prepare_datasets(self.filenames[0])
        if not self.train_only:
            self.test_dataloader = self.__prepare_datasets(self.filenames[2])
            if self.has_val:
                self.val_dataloader = self.__prepare_datasets(self.filenames[1])

        return self.train_dataloader, self.test_dataloader, self.val_dataloader

    def __read_and_sample(self, csv_path: str):
        # Returning  Specific Data Samples
        data = pd.read_csv(csv_path)
        sample_data = data.sample(frac=self.sample_fraction, random_state=42)
        return sample_data

    @staticmethod
    def __padding_size(dataset: list) -> int:
        # Returning  Max Size (Padding Size)
        return max([data.size(-1) for data in dataset])

    @staticmethod
    def __padding(data: list, padding_size: int) -> list:
        # Adds Pads to Tensors
        data_list = []
        for tensors in data:
            padding = (0, padding_size - tensors.size(-1))
            padded_tensor = F.pad(tensors, padding, mode='constant', value=0)
            data_list.append(padded_tensor)
        return data_list

    def __prepare_datasets(self, filename: str):
        # Preliminary Datasets
        if self.process is not "STT":
            csv_file = os.path.join(self.root_dir, filename)
        else:
            path_file = os.path.join(self.root_dir, list(self.labels.keys())[self.language])
            csv_file = os.path.join(str(path_file), filename)

        dataframe = self.__read_and_sample(str(csv_file))

        # Processing
        x_list = []
        y_list = []
        for index, data in dataframe.iterrows():
            if self.process == "VAD" or self.process == "LD":
                processed_data = self.__vad_ld(data)
            elif self.process == "STT":
                processed_data = self.__stt(data)
            else:
                raise KeyError("Please Select Right Process.")
            x_list.append(processed_data[0])
            y_list.append(processed_data[1])

        if self.process == "STT":
            y_list = self.__padding(y_list, self.__padding_size(y_list))

        # Return Datasets to DataLoader
        padded_data = torch.cat(self.__padding(x_list, self.__padding_size(x_list)))
        data_label = torch.stack(y_list)
        tensor_datasets = TensorDataset(padded_data, data_label)
        dataloader = DataLoader(tensor_datasets, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def __vad_ld(self, data: pd.Series):
        # VAD and LD Processing
        # .load Warning Is Not Important
        audio, sample_rate = torchaudio.load(data[0], normalize=True)
        audio = audio.to(self.device)
        audio = torchaudio.transforms.MFCC(sample_rate)(audio)

        if self.process == "LD":
            label = one_hot(torch.tensor(data[1]), num_classes=2).float()
        elif self.process == "VAD":
            audio = torchaudio.transforms.AmplitudeToDB()(audio)
            label = torch.tensor(data[1], dtype=torch.float).unsqueeze(0).to(self.device)
        else:
            raise KeyError("Please Select Right Process.")

        self.input_size = audio.size(1)
        return audio, label

    def __stt_labels(self, label: str) -> torch.Tensor:
        # STT Labels Processing
        label = label.lower()
        label = re.sub(r'\d', '', label)
        label = re.sub(r'[^\w\s]', '', label)

        tensors = torch.zeros(len(label)+1).to(self.device)

        for index, c in enumerate(label):
            index = self.alphabet.index(c) + 2
            tensors[index] = torch.tensor(index)
        tensors[-1] = torch.tensor(self.EOS_token)
        return tensors.long()

    def __stt(self, data: pd.Series):
        # STT Processing
        # .load Warning Is Not Important
        audio, sample_rate = torchaudio.load(data[0], normalize=True)
        audio = audio.to(self.device)
        audio = torchaudio.transforms.MelSpectrogram(sample_rate)(audio)
        label_data = self.__stt_labels(data[1])
        self.input_size = audio.size(1)
        return audio, label_data
