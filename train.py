from pytorch_lightning import Trainer
from lightning_model import LorenzLightningModule
from lightning_data import LorenzDataModule
from model import MLPModel, LSTMModel

# 假設你已經有 states = generate_lorenz_data(...)

# train_split, val_split, test_split = 0.7, 0.15, 0.15
# N = len(states)
# train_data = states[:int(N*train_split)]
# val_data   = states[int(N*train_split):int(N*(train_split+val_split))]
# test_data  = states[int(N*(train_split+val_split)):]

dm = LorenzDataModule(nc_file='data/lorenz_data_r=100.nc', normalize=True)

# 選 MLP
mlp = MLPModel(input_dim=30, hidden_dim=64, output_dim=3)

# Naive training
naive_module = LorenzLightningModule(mlp, lr=1e-3, mode="naive")
trainer = Trainer(max_epochs=20, accelerator="cpu")
trainer.fit(naive_module, dm)
trainer.test(naive_module, dm)

# Physics-guided training
phy_module = LorenzLightningModule(mlp, lr=1e-3, lam=0.1, mode="physics")
trainer = Trainer(max_epochs=20, accelerator="cpu")
trainer.fit(phy_module, dm)
trainer.test(phy_module, dm)
