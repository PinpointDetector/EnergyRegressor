import glob
import torch
import random

import hydra
from omegaconf import DictConfig
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from source.dataset import CombinedDataset, ProjectionDataModule
from source.model import ClassifierProjectionCNN, RegressionCNN

# Source - https://stackoverflow.com/a
# Posted by shtse8
# Retrieved 2026-01-28, License - CC BY-SA 4.0
def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()


class EnergyRegressor(pl.LightningModule):
    def __init__(
        self,
        model,
        lr=1e-3,
        targets=("E_lep",),
        use_weights=False,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.targets = targets
        self.use_weights = use_weights

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        (zx, zy), targets = batch

        y = torch.stack(
        [
            targets[t].float()
            for t in self.targets
        ],
        dim=1,  # (batch, n targets)
    )
        y_hat = self((zx, zy))

        weights = targets["weight"]

        if self.use_weights:
            loss = weighted_mse_loss(y_hat, y, weights)
        else:   
            loss = F.mse_loss(y_hat, y)
        
        self.log("train_loss", loss, prog_bar=True)
        rmse = torch.sqrt(loss)
        self.log("train_rmse", rmse, prog_bar=True)


        return loss

    def validation_step(self, batch, batch_idx):
        (zx, zy), targets = batch

        y = torch.stack(
        [
            targets[t].float()
            for t in self.targets
        ],
        dim=1,  # (batch, n targets)
        )

        y_hat = self((zx, zy))
        loss = F.mse_loss(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)
        rel_err = (y_hat - y) / y
        self.log("val_rel_err_mean", rel_err.mean(), prog_bar=True)
        self.log("val_rel_err_std", rel_err.std(), prog_bar=True)


    def test_step(self, batch, batch_idx):
        (zx, zy), targets = batch
        y = torch.stack(
        [
            targets[t].float()
            for t in self.targets
        ],
        dim=1,  # (batch, n targets)
        )

        y_hat = self((zx, zy))
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss)
        


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    
    def on_train_epoch_end(self):
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)



def run_training(cfg: DictConfig):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = TensorBoardLogger(
        # save_dir=cfg.logging.log_dir,
        save_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
    )

    pt_files = glob.glob(cfg.data.datapath)
    assert len(pt_files) > 0, "No .pt files found" 

    datamodule = ProjectionDataModule(
        pt_files=pt_files,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )

    model = EnergyRegressor(
        model=RegressionCNN(
            feature_dim=cfg.model.feature_dim,
            num_targets=len(cfg.training.targets),
            fc_dims=cfg.model.fc_dims,
            conv_dims=cfg.model.conv_dims,
            kernel_size=cfg.model.kernel_size,
            padding=cfg.model.padding,
            dropout=cfg.model.dropout,
        ).to(device),
        lr=1e-3,
        targets=cfg.training.targets,
    )

    
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )  
    
    early_stop_cb = EarlyStopping(
    monitor=cfg.training.early_stopping.monitor,
    min_delta=cfg.training.early_stopping.min_delta,
    patience=cfg.training.early_stopping.patience,
    mode="min",
    verbose=True,
    )


    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        log_every_n_steps=50,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
