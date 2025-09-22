from pytorch_lightning import Trainer
from lightning_model import LorenzLightningModule
from lightning_data import LorenzDataModule
from model import MLPModel, LSTMModel
import numpy as np
import logging

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

r_naive_model_metrics = []
r_physics_model_metrics = []

R_values = [0.5, 1, 2, 5, 10, 28, 50, 100, 1000]
#R_values = [100, 1000]
for r in R_values:
    naive_model_metrics = {'rmse':[], 'ACC':[]}
    physics_model_metrics = {'rmse': [], 'ACC':[]}
    dm = LorenzDataModule(nc_file=f'data/lorenz_data_r={r}.nc', normalize=True)

    for _ in range(30):
        mlp = MLPModel(input_dim=30, hidden_dim=64, output_dim=3)

        # Naive training
        naive_module = LorenzLightningModule(mlp, lr=1e-3, mode="naive")
        trainer = Trainer(max_epochs=20, accelerator="cpu", enable_progress_bar=False, enable_model_summary=False, logger=None)
        trainer.fit(naive_module, dm)
        trainer.test(naive_module, dm)

        naive_model_metrics['rmse'].append(trainer.callback_metrics['test_rmse'])
        naive_model_metrics['ACC'].append(trainer.callback_metrics['test_ACC'])

        # Physics-guided training
        phy_module = LorenzLightningModule(mlp, lr=1e-3, lam=0.1, mode="energy change loss")
        trainer = Trainer(max_epochs=20, accelerator="cpu", enable_progress_bar=False, enable_model_summary=False, logger=None)
        trainer.fit(phy_module, dm)
        trainer.test(phy_module, dm)

        physics_model_metrics['rmse'].append(trainer.callback_metrics['test_rmse'])
        physics_model_metrics['ACC'].append(trainer.callback_metrics['test_ACC'])
    r_naive_model_metrics.append((naive_model_metrics['rmse'], naive_model_metrics['ACC']))
    r_physics_model_metrics.append((physics_model_metrics['rmse'], physics_model_metrics['ACC']))

print( "                       naive                        physics")
print( "                   rmse            ACC             rmse            ACC")
#print(f"r={r}    (X.XXXX±X.XXXX, X.XXXX±X.XXXX)  (X.XXXX±X.XXXX, X.XXXX±X.XXXX)")
for i, r in enumerate(R_values):
    naive_rmse_mean = np.mean(r_naive_model_metrics[i][0])
    naive_rmse_std = np.std(r_naive_model_metrics[i][0])
    naive_ACC_mean = np.mean(r_naive_model_metrics[i][1])
    naive_ACC_std = np.std(r_naive_model_metrics[i][1])

    physics_rmse_mean = np.mean(r_physics_model_metrics[i][0])
    physics_rmse_std = np.std(r_physics_model_metrics[i][0])
    physics_ACC_mean = np.mean(r_physics_model_metrics[i][1])
    physics_ACC_std = np.std(r_physics_model_metrics[i][1])

    print(f"r={r:.2f}    ({naive_rmse_mean:.4f}±{naive_rmse_std:.4f}, {naive_ACC_mean:.4f}±{naive_ACC_std:.4f})  ({physics_rmse_mean:.4f}±{physics_rmse_std:.4f}, {physics_ACC_mean:.4f}±{physics_ACC_std:.4f})")