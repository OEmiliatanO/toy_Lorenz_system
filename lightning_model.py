import pytorch_lightning as pl
import torch

def physics_loss(pred, true, lam=0.1):
    mse = torch.mean((pred - true)**2)
    energy_pred = torch.sum(pred**2, dim=1)
    energy_true = torch.sum(true**2, dim=1)
    energy_mse = torch.mean((energy_pred - energy_true)**2)
    return mse + lam * energy_mse

class LorenzLightningModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3, lam=0.1, mode="naive"):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lam = lam
        self.mode = mode
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
        else:  # physics guided
            loss = physics_loss(y_hat, y, lam=self.lam)
        rmse_val = self.rmse(y_hat, y[:, -1])
        acc_val = self.acc(y_hat, y[:, -1])
        self.log(f"train_loss", loss)
        self.log(f"train_rmse", rmse_val)
        self.log(f"train_ACC", acc_val)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        rmse = torch.sqrt(torch.mean((y_hat - y)**2))
        self.log("val_rmse", rmse)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        rmse = torch.sqrt(torch.mean((y_hat - y)**2))
        self.log("test_rmse", rmse)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
