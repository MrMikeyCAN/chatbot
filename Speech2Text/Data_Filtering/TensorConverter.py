import torch
from torch.functional import F
import torchaudio
import warnings
import pandas as pd
import json
import os
from torch.utils.data import TensorDataset, DataLoader
import subprocess
import re

warnings.filterwarnings("ignore")

"""
0: Vad
1: Language_Detection
2: Speech_to_Text
3: Noise_Filter
"""

class TensorConverter:
    def __init__(self, sample_fraction: float, batch_size: int,
                 process_index: int = 0, json_filename: str = "data.json",
                 python_file: str = "Split_Datasets", filetype: str = None,
                 language: int = None, alphabet: str = None):

        # Running Split Python File
        python_file += ".py"
        subprocess.run(["python", python_file])

        # Parameters
        self.sample_fraction = sample_fraction
        self.batch_size = batch_size
        self.process_index = process_index
        self.filetype = filetype if filetype is not None else ".csv"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reading json file
        try:
            with open(json_filename, 'r', encoding='utf-8') as f:
                json_file = json.load(f)
        except KeyError:
            raise KeyError("Please Control Json File.")

        # Process Control
        has_process = False
        for index, model in enumerate(json_file):
            if index == self.process_index:
                json_file = json_file[model]
                has_process = True
                break

        if not has_process:
            raise ValueError("Do Not Have Process Parameters")

        # Getting Items
        self.train_only = json_file['train_only']
        self.has_val = json_file['has_val']
        self.root_dir = json_file['root_dir']
        self.train_filename = json_file['train_filename']
        self.val_filename = json_file['val_filename']
        self.test_filename = json_file['test_filename']
        self.labels = json_file['labels']

        if self.process_index == 2:
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

        # For STT Labels
        if alphabet is not None:
            self.alphabet = alphabet
        else:
            self.alphabet = "abcdefghijklmnopqrstuvwxyzçşıüöğ "

        self.SOS_token = 0
        self.EOS_token = 1

    def process_data(self):
        # Processing Dataloaders
        self.train_dataloader = self.__prepare_datasets(self.train_filename)
        if not self.train_only:
            self.test_dataloader = self.__prepare_datasets(self.test_filename)
            if self.has_val:
                self.val_dataloader = self.__prepare_datasets(self.val_filename)

        return self.train_dataloader, self.test_dataloader, self.val_dataloader

    def __read_and_sample(self, csv_path: str):
        # Returning  Specific Data Samples
        data = pd.read_csv(csv_path + self.filetype)
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
        for tensor in data:
            padding = (0, padding_size - tensor.size(-1))
            padded_tensor = F.pad(tensor, padding, mode='constant', value=0)
            data_list.append(padded_tensor)
        return data_list

    def __prepare_datasets(self, filename: str):
        # Preliminary Datasets
        if self.process_index != 2:
            csv_file = os.path.join(self.root_dir, filename)
        else:
            path_file = os.path.join(self.root_dir, list(self.labels.keys())[self.language])
            csv_file = os.path.join(str(path_file), filename)

        data = self.__read_and_sample(str(csv_file))

        if self.process_index == 0 or self.process_index == 1:
            processed_data = self.__vad_ld(data)
        elif self.process_index == 2:
            processed_data = self.__stt(data)
        else:
            raise KeyError("Please Select Right Process Index.")

        # Return Datasets to DataLoader
        padded_data = torch.cat(self.__padding(processed_data[0], self.__padding_size(processed_data[0])))
        data_label = torch.stack(processed_data[1])
        tensor_datasets = TensorDataset(padded_data, data_label)
        dataloader = DataLoader(tensor_datasets, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def __vad_ld(self, dataframe: pd.DataFrame):
        # VAD and LD Processing
        x_list = []
        y_list = []
        for index, data in dataframe.iterrows():
            # .load warning is not important
            audio, sample_rate = torchaudio.load(data[0], normalize=True)
            audio = audio.to(self.device)
            audio = torchaudio.transforms.MFCC(sample_rate)(audio)
            if self.process_index == 1:
                audio = torchaudio.transforms.AmplitudeToDB()(audio)
            x_list.append(audio)
            y_list.append(torch.tensor(data[1]).unsqueeze(0).to(self.device))
        return x_list, y_list

    def __stt_labels(self, label: str) -> torch.Tensor:
        # STT Labels Processing
        label = label.lower()
        label = re.sub(r'\d', '', label)
        label = re.sub(r'[^\w\s]', '', label)

        tensor = torch.zeros(len(label)+1).to(self.device)

        for i, c in enumerate(label):
            index = self.alphabet.index(c) + 2
            tensor[i] = torch.tensor([index])
        tensor[-1] = torch.tensor([self.EOS_token])
        return tensor.unsqueeze(0).long()

    def __stt(self, dataframe: pd.DataFrame):
        # STT Processing
        x_list = []
        y_list = []
        for index, data in dataframe.iterrows():
            # .load warning is not important
            audio, sample_rate = torchaudio.load(data[0], normalize=True)
            audio = audio.to(self.device)
            audio = torchaudio.transforms.MelSpectrogram(sample_rate)(audio)
            x_list.append(audio)
            label_data = self.__stt_labels(data[1])
            y_list.append(label_data)
        y_list = self.__padding(y_list, self.__padding_size(y_list))
        return x_list, y_list
