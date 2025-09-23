import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn.functional import cosine_similarity, mse_loss

def Lorenz_diff(X):
    x, y, z = X[:,0], X[:,1], X[:,2]
    dx = 10 * (y - x)
    dy = x * (1 - z) - y
    dz = x * y - (8/3) * z
    dX = torch.stack([dx, dy, dz], dim=1)
    return dX

def energy_change_loss(pred, pre_gt, gt, dt):
    # energy_pred = torch.sum(pred**2, dim=1)
    # energy_true = torch.sum(true**2, dim=1)
    # energy_mse = torch.mean((energy_pred - energy_true)**2)
    # return energy_mse
    pred_dX = (pred-pre_gt)/dt
    true_dX = Lorenz_diff(pre_gt)
    loss = torch.mean((2 * torch.sum(gt * pred_dX, dim=1) - 2 * torch.sum(gt * true_dX, dim=1))**2)
    return loss

def ODE_loss(pred, pre_gt, dt):
    pred_dX = (pred-pre_gt)/dt
    true_dX = Lorenz_diff(pre_gt)
    loss = torch.mean((true_dX-pred_dX)**2)
    return loss

class LorenzLightningModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3, lam=1, mode="naive"):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lam = lam
        self.mode = mode
        self.dt=0.01
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.acc = torchmetrics.CosineSimilarity()
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch # [batch size, ]
        y_hat = self(x)
        if self.mode == "naive":
            loss = torch.mean((y_hat - y)**2)
        elif self.mode == "ODE loss":
            loss = torch.mean((y_hat - y)**2) + self.lam * ODE_loss(y_hat, x[:,-1], self.dt)
        elif self.mode == "energy change loss":
            loss = torch.mean((y_hat - y)**2) + self.lam * energy_change_loss(y_hat, x[:,-1], y, self.dt)
        rmse_val = torch.mean(torch.sqrt(mse_loss(y_hat, y)))
        acc_val = torch.mean(cosine_similarity(y_hat, y, dim=1))
        self.log(f"train_loss", loss)
        self.log(f"train_rmse", rmse_val)
        self.log(f"train_ACC", acc_val)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        rmse = torch.mean(torch.sqrt(mse_loss(y_hat, y))) #torch.sqrt(torch.mean((y_hat - y)**2))
        acc = torch.mean(cosine_similarity(y_hat, y, dim=1))
        self.log("val_rmse", rmse)
        self.log("val_ACC", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        rmse = torch.mean(torch.sqrt(mse_loss(y_hat, y))) #rmse = torch.sqrt(torch.mean((y_hat - y)**2))
        acc = torch.mean(cosine_similarity(y_hat, y, dim=1))
        self.log("test_rmse", rmse)
        self.log("test_ACC", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
