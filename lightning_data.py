import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import xarray as xr

class LorenzDataset(Dataset):
    def __init__(self, data, seq_len=10):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return x, y

class LorenzDataModule(pl.LightningDataModule):
    def __init__(self, nc_file="lorenz.nc", seq_len=10, batch_size=32):
        super().__init__()
        ds = xr.open_dataset(nc_file)
        self.data = ds["state"].values
        self.seq_len = seq_len
        self.batch_size = batch_size

    def setup(self, stage=None):
        N = len(self.data)
        train_size = int(N*0.8)
        self.train_dataset = LorenzDataset(self.data[:train_size], self.seq_len)
        self.val_dataset = LorenzDataset(self.data[train_size:], self.seq_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)