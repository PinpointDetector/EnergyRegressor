import argparse
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import os
import logging

from source.model import RegressionCNN
from source.train import ProjectionDataModule   
from source.train import EnergyRegressor
from source.preprocess import CNNProjectionDataset


def load_run_config(run_dir: str):

    if not os.path.isdir(run_dir):
        logging.error(f"Run directory {run_dir} does not exist")
        raise ValueError(f"Run directory {run_dir} does not exist")

    cfg_path = Path(run_dir) / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)

    logging.info(f"Loaded config from {cfg_path}")
    return cfg


def get_checkpoint_path(cfg: DictConfig):

    # Load a specific checkpoint if provided
    if cfg.testing.checkpoint_filepath is not None:
        if not os.path.isfile(cfg.testing.checkpoint_filepath):
            logging.error(f"Checkpoint file {cfg.testing.checkpoint_filepath} does not exist")
            raise ValueError(f"Checkpoint file {cfg.testing.checkpoint_filepath} does not exist")
        
        logging.info(f"Using checkpoint: {cfg.testing.checkpoint_filepath}")
        return cfg.testing.checkpoint_filepath
    
    # Otherwise, find the latest checkpoint in the run directory
    ckpt_dir = os.path.join(cfg.testing.run_dir, cfg.testing.checkpoint_dir, "*.ckpt")
    logging.info(f"Searching for checkpoints in directory: {ckpt_dir}")
    checkpoint_paths = glob.glob(ckpt_dir)
    
    if len(checkpoint_paths) == 0:
        logging.error("No checkpoint found in run directory")
        raise ValueError("No checkpoint found in run directory")
    elif len(checkpoint_paths) > 1:
        logging.warning("Multiple checkpoints found, using the first one")
    
    checkpoint_path = checkpoint_paths[0]

    logging.info(f"Using checkpoint: {checkpoint_path}")
    return checkpoint_path


@torch.no_grad()
def run_inference(cfg, model, dataloader, device):
    model.eval()
    model.freeze()

    targets_true = {t : [] for t in cfg.training.targets}
    targets_pred = {t : [] for t in cfg.training.targets}

    for batch in dataloader:
        (zx, zy), targets = batch

        zx = zx.to(device)
        zy = zy.to(device)


        y = torch.stack(
        [
            targets[t].float()
            for t in cfg.training.targets
        ],
        dim=1,  # (batch, n targets)
        )
        
        y_pred = model((zx, zy))

        for i, t in enumerate(cfg.training.targets):
            targets_true[t].append(y.cpu()[:,i].numpy())
            targets_pred[t].append(y_pred.cpu()[:,i].numpy())

    for t in cfg.training.targets:
        targets_true[t] = np.concatenate(targets_true[t])
        targets_pred[t] = np.concatenate(targets_pred[t])


    # Apply any necessary inverse transformations here
    for t in cfg.training.targets:
        if cfg.variables.get(t, None):
            if cfg.variables[t].get("transformation", None) == "log10":
                targets_true[t] = 10 ** targets_true[t]
                targets_pred[t] = 10 ** targets_pred[t]
            elif cfg.variables[t].get("transformation", None) == "eta":
                targets_true[t] = 2 * np.arctan(np.exp(targets_true[t])) - np.pi/2
                targets_pred[t] = 2 * np.arctan(np.exp(targets_pred[t])) - np.pi/2


    return targets_true, targets_pred


def plot_resolution_hists(cfg, varname, targets_true, targets_pred):

    var_cfg = cfg.variables.get(varname, {})

    assert var_cfg is not None, f"No config found for variable {varname}"
    assert "bins" in var_cfg, f"No binning defined for variable {varname} in config"

    y_true = targets_true[varname]
    y_pred = targets_pred[varname]

    resolution = (y_pred - y_true) / y_true

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    axes = axes.flatten()

    bin_labels = []
    for edgemin, edgemax in var_cfg.bins:
        if edgemax == np.inf:
            bin_labels.append(f"> {edgemin} {var_cfg.get("unit", "")}")
        else:
            bin_labels.append(f"{edgemin}–{edgemax} {var_cfg.get("unit", "")}")

    for i, ((ymin, ymax), label) in enumerate(zip(var_cfg.bins, bin_labels)):
        try:
            ax = axes[i]
        except IndexError:
            logging.warning("More bins defined than subplots available")
            break

        mask = (y_true >= ymin) & (y_true < ymax)
        res_bin = resolution[mask]

        # if len(res_bin) < 50:
        #     ax.text(0.5, 0.5, "Too few events", ha="center", va="center")
        #     continue

        ax.hist(
            res_bin,
            bins=60,
            range=(-1, 1),
            histtype="step",
            linewidth=1.5,
        )

        mean = np.mean(res_bin)
        sigma = np.std(res_bin)

        ax.set_title(label)
        latex_var = var_cfg.get("latex", varname)
        ax.set_xlabel(rf"$({latex_var}^{{\rm reco}} - {latex_var}^{{\rm true}}) / {latex_var}^{{\rm true}}$")
        ax.set_ylabel("Events")

        ax.text(
            0.05, 0.95,
            f"Mean = {mean:.3f}\nStd. = {sigma:.3f}",
            transform=ax.transAxes,
            va="top",
        )

    fig.delaxes(axes[-1])

    outfile = os.path.join(cfg.testing.run_dir, f"{varname}_ResolutionPerBin")
    for fmt in cfg.plotting.formats:
        extn = fmt.lower().replace(".", "")
        plt.savefig(f"{outfile}.{extn}", dpi=300)
    logging.info(f"Plotted {outfile}")


def plot_true_vs_reco(cfg, varname, targets_true, targets_pred, logscale=False):

    var_cfg = cfg.variables.get(varname, {})

    assert var_cfg is not None, f"No config found for variable {varname}"

    y_true = targets_true[varname]
    y_pred = targets_pred[varname]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Log-spaced bins work best here
    if logscale:
        bins = np.logspace(
            np.log10(min(y_true.min(), y_pred.min())),
            np.log10(max(y_true.max(), y_pred.max())),
            100,
        )
    else:
        bins = np.linspace(
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max()),
            100,
        )

    h = ax.hist2d(
        y_true,
        y_pred,
        bins=[bins, bins],
        norm="log",
        cmap="viridis",
    )

    plt.colorbar(h[3], label="Events", ax=ax)

    # y = x reference line
    if logscale:
        x = np.logspace(
            np.log10(y_true.min()),
            np.log10(y_true.max()),
            100,
        )
    else:
        x = np.linspace(
            y_true.min(),
            y_true.max(),
            100,
        )
            
    ax.plot(x, x, "r--", linewidth=1, label=rf"${var_cfg.get("latex", varname)}^{{\rm reco}} = {var_cfg.get("latex", varname)}^{{\rm true}}$")

    if logscale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel(rf"True ${var_cfg.get("latex", varname)}$ {var_cfg.get("unit", "")}")
    ax.set_ylabel(rf"Reconstructed ${var_cfg.get("latex", varname)}$ {var_cfg.get("unit", "")}")
    ax.legend()
    plt.tight_layout()
    outfile = os.path.join(cfg.testing.run_dir, f"{varname}_TrueVsReco")
    for fmt in cfg.plotting.formats:
        extn = fmt.lower().replace(".", "")
        plt.savefig(f"{outfile}.{extn}", dpi=300)
    logging.info(f"Plotted {outfile}")
    

def plot_resolution_vs_target(cfg, varname, targets_true, targets_pred):

    var_cfg = cfg.variables.get(varname, {})

    assert var_cfg is not None, f"No config found for variable {varname}"
    assert "bins" in var_cfg, f"No binning defined for variable {varname} in config"

    y_true = targets_true[varname]
    y_pred = targets_pred[varname]

    resolution = (y_pred - y_true) / y_true

    bin_centers = []
    sigmas = []
    sigma_errs = []

    for emin, emax in var_cfg.bins:
        mask = (y_true >= emin) & (y_true < emax)
        res_bin = resolution[mask]

        if len(res_bin) < 50:
            continue

        sigma = np.std(res_bin)
        sigma_err = sigma / np.sqrt(2 * len(res_bin))
        center = np.mean(y_true[mask])

        bin_centers.append(center)
        sigmas.append(sigma)
        sigma_errs.append(sigma_err)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        bin_centers,
        sigmas,
        yerr=sigma_errs,
        fmt="o",
    )
    ax.set_xscale("log")
    latex_var = var_cfg.get("latex", varname)
    unit = var_cfg.get("unit", "")
    ax.set_xlabel(rf"True ${latex_var}$ {unit}")
    ax.set_ylabel(rf"${latex_var}$ Resolution (σ)")
    outfile = os.path.join(cfg.testing.run_dir, f"{varname}_ResolutionVsTrue")
    for fmt in cfg.plotting.formats:
        extn = fmt.lower().replace(".", "")
        plt.savefig(f"{outfile}.{extn}", dpi=300)
    logging.info(f"Plotted {outfile}")


def run_testing(cfg: DictConfig):

    logging.info("Starting testing...")

    # Load the config for this run
    cfg_this_run = load_run_config(cfg.testing.run_dir)
    
    device = torch.device("cuda" if cfg.device == "gpu" else "cpu")
    
    # Get the checkpoint path
    checkpoint_path = get_checkpoint_path(cfg)

    # Load the dataset
    pt_files = glob.glob(cfg_this_run.data.datapath)
    assert len(pt_files) > 0, "No .pt files found"

    datamodule = ProjectionDataModule(
        pt_files=pt_files,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )

    # Reconstruct model exactly as in training
    backbone=RegressionCNN(
            feature_dim=cfg_this_run.model.feature_dim,
            num_targets=len(cfg_this_run.training.targets),
            fc_dims=cfg_this_run.model.fc_dims,
            conv_dims=cfg_this_run.model.conv_dims,
            kernel_size=cfg_this_run.model.kernel_size,
            padding=cfg_this_run.model.padding,
            dropout=cfg_this_run.model.dropout,
        ).to(device)
    model = EnergyRegressor.load_from_checkpoint(
        checkpoint_path,
        model=backbone,
    )
    model.to(device)


    targets_true, targets_pred = run_inference(
        cfg_this_run,
        model,
        datamodule.test_dataloader(),
        device,
    )

    for varname in cfg_this_run.training.targets:

        plot_resolution_hists(cfg, varname, targets_true, targets_pred)
        plot_resolution_vs_target(cfg, varname, targets_true, targets_pred)
        plot_true_vs_reco(cfg, varname, targets_true, targets_pred)