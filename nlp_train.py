import pandas as pd
from utils import tokenize, lemma
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

### * Loading and preprocecing the dataset for language detecetion
LD_dataset = pd.read_csv("LD.csv")

X_LD = LD_dataset.iloc[:, 0].values
y_LD = LD_dataset.iloc[:, 1].values

X_LD = [lemma(tokenize(word)) for word in X_LD if isinstance(word, str)]


X_LD = sorted(X_LD)

### * Splitting the dataset
X_LD_train, X_LD_test, y_LD_train, y_LD_test = train_test_split(
    X_LD, y_LD, test_size=0.2, random_state=42
)


### * Settings
epochs = 500
batch_size = 8
learning_rate = 0.01
hidden_size = 8
input_LD_size = len(X_LD)
output_LD_size = 17


### ! Pytorch model
class ChatDataset(Dataset):
    def __init__(self, X_indices, y_data):
        self.n_samples = len(X_indices)
        self.x_data = X_indices
        self.y_data = y_data

    def __getitem__(self, index):
        return torch.tensor(self.x_data[index]), torch.tensor(self.y_data[index])

    def __len__(self):
        return self.n_samples