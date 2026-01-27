import glob
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import random
from torch.utils.data import Dataset
from preprocess import CNNProjectionDataset
from model import ClassifierProjectionCNN, RegressionCNN
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from model import RegressionCNN  # your model file

from test_primLep import *

class CombinedDataset(Dataset):
    def __init__(self, pt_files):
        self.datasets = [torch.load(f, map_location="cpu") for f in pt_files]
        self.cum_lengths = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self.cum_lengths.append(total)

    def __len__(self):
        return self.cum_lengths[-1]

    def __getitem__(self, idx):
        for ds_idx, end in enumerate(self.cum_lengths):
            if idx < end:
                start = 0 if ds_idx == 0 else self.cum_lengths[ds_idx - 1]
                return self.datasets[ds_idx][idx - start]
        raise IndexError


class ProjectionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        pt_files,
        batch_size=16,
        num_workers=4,
    ):
        super().__init__()
        self.pt_files = pt_files
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.setup()

    def setup(self, stage=None):
        files = list(self.pt_files)
        random.shuffle(files)

        n = len(files)
        n_test = int(0.2 * n)
        n_val = int(0.1 * n)

        test_files = files[:n_test]
        val_files = files[n_test:n_test + n_val]
        train_files = files[n_test + n_val:]

        self.train_ds = CombinedDataset(train_files)
        self.val_ds   = CombinedDataset(val_files)
        self.test_ds  = CombinedDataset(test_files)

        print(f"Train events: {len(self.train_ds)}")
        print(f"Val events:   {len(self.val_ds)}")
        print(f"Test events:  {len(self.test_ds)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class EnergyRegressor(pl.LightningModule):
    def __init__(
        self,
        model,
        lr=1e-3,
        targets=("E_lep",)
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.targets = targets

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

        plot_resolution_hists(y.cpu(), y_hat.cpu())
        plot_resolution_vs_energy(y.cpu(), y_hat.cpu())
        plot_true_vs_reco(y.cpu(), y_hat.cpu())


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



if __name__ == "__main__":

    logger = TensorBoardLogger(
    save_dir="logs",
    name="energy_regression_primLep",
)


    pt_files = glob.glob("*.pt")

    datamodule = ProjectionDataModule(
        pt_files=pt_files,
        batch_size=64,
        num_workers=16,
    )

    model = EnergyRegressor(
        model=RegressionCNN(feature_dim=128, num_targets=2),
        lr=1e-3,
        targets=("E_lep", "Eta_lep")
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )  
    
    early_stop_cb = EarlyStopping(
    monitor="val_loss",
    min_delta=0.0,
    patience=50,          # stop after 8 epochs without improvement
    mode="min",
    verbose=True,
    )


    # trainer = Trainer(max_epochs=100, overfit_batches=1)
    # model.eval()
    # (zx, zy), targets = datamodule.val_ds[0]
    # pred = model((zx.unsqueeze(0), zy.unsqueeze(0)))
    # print(pred.item(), targets["E_lep"])


    trainer = Trainer(
        max_epochs=500,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        log_every_n_steps=50,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


    