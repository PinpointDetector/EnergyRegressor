#!/usr/bin/env python3

import argparse
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

import pytorch_lightning as pl

# ---------------------------
# Your project imports
# ---------------------------
from model import RegressionCNN
from train import ProjectionDataModule   # or wherever you defined it
from train import EnergyRegressor              # LightningModule
from preprocess import CNNProjectionDataset


# ---------------------------
# Energy binning (GeV)
# ---------------------------
ENERGY_BINS = [
    (0, 100),
    (100, 200),
    (200, 300),
    (300, 400),
    (400, 600),
    (600, 1000),
    (1000, np.inf),
]

BIN_LABELS = [
    "0–100 GeV",
    "100–200 GeV",
    "200–300 GeV",
    "300–400 GeV",
    "400–600 GeV",
    "600–1000 GeV",
    "> 1000 GeV",
]


# ---------------------------
# Argument parser
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-bin energy resolution plots"
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to Lightning checkpoint (.ckpt)",
    )

    parser.add_argument(
        "--pt-files",
        default="*.pt",
        help="Glob for .pt dataset files (default: *.pt)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="gpu" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


# ---------------------------
# Inference
# ---------------------------
@torch.no_grad()
def run_inference(model, dataloader, device):
    model.eval()
    model.freeze()

    E_true = []
    E_pred = []

    for batch in dataloader:
        (zx, zy), targets = batch

        zx = zx.to(device)
        zy = zy.to(device)

        y_log_true = targets["E_lep"].float().to(device)
        y_log_pred = model((zx, zy))

        # Undo log10 scaling
        E_true.append(10 ** y_log_true.cpu().numpy())
        E_pred.append(10 ** y_log_pred.cpu().numpy())

    return (
        np.concatenate(E_true),
        np.concatenate(E_pred),
    )


# ---------------------------
# Plotting
# ---------------------------
def plot_resolution_hists(E_true, E_pred):
    resolution = (E_pred - E_true) / E_true

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    axes = axes.flatten()

    for i, ((emin, emax), label) in enumerate(zip(ENERGY_BINS, BIN_LABELS)):
        ax = axes[i]

        mask = (E_true >= emin) & (E_true < emax)
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
        ax.set_xlabel(r"$(E_{\rm reco} - E_{\rm true}) / E_{\rm true}$")
        ax.set_ylabel("Events")

        ax.text(
            0.05, 0.95,
            f"Mean = {mean:.3f}\nσ = {sigma:.3f}",
            transform=ax.transAxes,
            va="top",
        )

    fig.delaxes(axes[-1])
    plt.savefig("EnergyResolutionPerBin_E_lep.pdf")
    # plt.show()
    print("Plotted EnergyResolutionPerBin_E_lep.pdf")


def plot_true_vs_reco(E_true, E_pred, logscale=False):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Log-spaced bins work best here
    if logscale:
        bins = np.logspace(
            np.log10(min(E_true.min(), E_pred.min())),
            np.log10(max(E_true.max(), E_pred.max())),
            100,
        )
    else:
        bins = np.linspace(
            min(E_true.min(), E_pred.min()),
            max(E_true.max(), E_pred.max()),
            100,
        )

    h = ax.hist2d(
        E_true,
        E_pred,
        bins=[bins, bins],
        norm="log",
        cmap="viridis",
    )

    plt.colorbar(h[3], label="Events", ax=ax)

    # y = x reference line
    if logscale:
        x = np.logspace(
            np.log10(E_true.min()),
            np.log10(E_true.max()),
            100,
        )
    else:
        x = np.linspace(
            E_true.min(),
            E_true.max(),
            100,
        )
            
    ax.plot(x, x, "r--", linewidth=1, label=r"$E_{\rm reco} = E_{\rm true}$")

    if logscale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("True Energy [GeV]")
    ax.set_ylabel("Reconstructed Energy [GeV]")
    ax.legend()
    plt.tight_layout()
    plt.savefig("ETrueVsEReco_E_lep.pdf")
    print("Plotted ETrueVsEReco_E_lep.pdf")
    


def plot_resolution_vs_energy(E_true, E_pred):
    resolution = (E_pred - E_true) / E_true

    bin_centers = []
    sigmas = []
    sigma_errs = []

    for emin, emax in ENERGY_BINS:
        mask = (E_true >= emin) & (E_true < emax)
        res_bin = resolution[mask]

        if len(res_bin) < 50:
            continue

        sigma = np.std(res_bin)
        sigma_err = sigma / np.sqrt(2 * len(res_bin))
        center = np.mean(E_true[mask])

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
    ax.set_xlabel("True Energy [GeV]")
    ax.set_ylabel("Energy Resolution (σ)")
    plt.savefig("EnergyResolution_E_lep.pdf")
    print("Plotted EnergyResolution_E_lep.pdf")


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()

    pt_files = glob.glob(args.pt_files)
    assert len(pt_files) > 0, "No .pt files found"

    datamodule = ProjectionDataModule(
        pt_files=pt_files,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup()

    # Reconstruct model exactly as in training
    backbone = RegressionCNN(feature_dim=128)
    model = EnergyRegressor.load_from_checkpoint(
        args.checkpoint,
        model=backbone,
    )

    device = torch.device("cuda" if args.device == "gpu" else "cpu")
    model.to(device)

    E_true, E_pred = run_inference(
        model,
        datamodule.test_dataloader(),
        device,
    )

    plot_resolution_hists(E_true, E_pred)
    plot_resolution_vs_energy(E_true, E_pred)
    plot_true_vs_reco(E_true, E_pred)



if __name__ == "__main__":
    main()
