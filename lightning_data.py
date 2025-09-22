import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import xarray as xr
import numpy as np

class LorenzDataset(Dataset):
    def __init__(self, data, seq_len=10, mean=None, std=None, normalize=False):
        self.seq_len = seq_len
        self.normalize = normalize
        self.data = torch.tensor(data, dtype=torch.float32)
        if normalize and mean is not None and std is not None:
            self.mean = torch.tensor(mean, dtype=torch.float32)
            self.std = torch.tensor(std, dtype=torch.float32)
            self.data = (self.data - self.mean) / (self.std + 1e-8)

        self.X, self.Y = self.create_sequences(data, seq_len)

    def create_sequences(self, states, seq_len):
        X, Y = [], []
        for i in range(len(states) - seq_len):
            X.append(states[i:i+seq_len])
            Y.append(states[i+seq_len])
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class LorenzDataModule(pl.LightningDataModule):
    def __init__(self, nc_file="lorenz.nc", seq_len=10, batch_size=32, normalize=False):
        super().__init__()
        ds = xr.open_dataset(nc_file)
        self.data = ds["state"].values
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.normalize = normalize

    def setup(self, stage=None):
        N = len(self.data)
        train_size = int(N*0.8)
        test_size = N - train_size

        if self.normalize:
            train_data = self.data[:train_size-64]
            self.mean = train_data.mean(axis=0)
            self.std = train_data.std(axis=0)

        self.train_dataset = LorenzDataset(self.data[:train_size-64], self.seq_len, mean=self.mean, std=self.std, normalize=self.normalize)
        self.val_dataset = LorenzDataset(self.data[train_size-64:train_size], self.seq_len, mean=self.mean, std=self.std, normalize=self.normalize)
        self.test_dataset = LorenzDataset(self.data[train_size:], self.seq_len, mean=self.mean, std=self.std, normalize=self.normalize)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)