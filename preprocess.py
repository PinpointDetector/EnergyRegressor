import logging

import hist
import numpy as np
import pandas as pd
import torch
import glob
from torch.utils.data import Dataset
import uproot
import os
from tqdm import tqdm
import awkward as ak
import matplotlib.pyplot as plt
import mplhep

import numpy as np

def compute_energy_weights(energies, n_bins=20):
    logE = np.log10(energies)

    bins = np.linspace(logE.min(), logE.max(), n_bins + 1)
    bin_idx = np.digitize(logE, bins) - 1

    counts = np.bincount(bin_idx, minlength=n_bins)
    counts[counts == 0] = 1  # avoid div by zero

    weights = 1.0 / counts[bin_idx]
    weights /= weights.mean()  # normalize

    return weights


def create_projections_fast(hit_layer, hit_col, hit_row, bins):

    mask = hit_layer >= 4
    if not np.any(mask):
        return None, None, None, None

    hit_layer = hit_layer[mask]
    hit_col   = hit_col[mask]
    hit_row   = hit_row[mask]

    x_half = bins[0] // 2
    y_half = bins[1] // 2

    mean_x = hit_col.mean(dtype=np.float64)
    mean_y = hit_row.mean(dtype=np.float64)

    zx, _, _ = np.histogram2d(
        hit_layer,
        hit_col,
        bins=(bins[2], bins[0]),
        range=((0, bins[2]), (mean_x - x_half, mean_x + x_half)),
    )

    zy, _, _ = np.histogram2d(
        hit_layer,
        hit_row,
        bins=(bins[2], bins[1]),
        range=((0, bins[2]), (mean_y - y_half, mean_y + y_half)),
    )

    return int(mean_x), int(mean_y), zx, zy



class CNNProjectionDataset(Dataset):
    def __init__(self, file_name, bins):
        root_file = uproot.open(file_name)

        truth = root_file["event"].arrays(library="np")
        primaries = root_file["primaries"].arrays(library="np")
        hits = root_file["Hits/pixelHits"]

        event_id   = hits["event_id"].array(library="np")
        hit_layer  = hits["hit_layerID"].array(library="np")
        hit_col    = hits["hit_colID"].array(library="np")
        hit_row    = hits["hit_rowID"].array(library="np")


        # Build truth lookup
        truth_map = {
            evt: (E, x, y, z)
            for evt, E, x, y, z in zip(
                truth["evtID"],
                truth["initE"],
                truth["initX"],
                truth["initY"],
                truth["initZ"],
            )
        }

        # Build primary lepton lookup
        lepton_mask = np.isin(np.abs(primaries["PDG"]), [11, 13, 15])
        lepton_mask = np.isin(np.abs(primaries["PDG"]), [11, 13, 15])

        primaries = {
            key: value[lepton_mask]
            for key, value in primaries.items()
        }

        primaries_map = {
            evt: (E, theta)
                for evt, E, theta in zip(
                    primaries["evtID"],
                    primaries["E"],
                    primaries["Eta"]
                )
        }

        self.zx = []
        self.zy = []
        self.nhits = []
        self.targets = []
        self.sample_weights = compute_energy_weights(truth["initE"])

        # Group hits ONCE
        unique_evt, evt_index = np.unique(event_id, return_inverse=True)

        for i, evt in tqdm(enumerate(unique_evt), total=len(unique_evt)):
            idx = np.where(evt_index == i)[0]

            if evt not in truth_map:
                continue

            mean_x, mean_y, zx, zy = create_projections_fast(
                np.ravel(hit_layer[idx][0]),
                np.ravel(hit_col[idx][0]),
                np.ravel(hit_row[idx][0]),
                bins,
            )
            if zx is None:
                continue
            nhits = len(np.ravel(hit_row[idx][0]))

            zx = torch.from_numpy(np.log1p(zx)).unsqueeze(0).float()
            zy = torch.from_numpy(np.log1p(zy)).unsqueeze(0).float()

            nhits = torch.from_numpy(np.log1p([nhits])).unsqueeze(0).float()

            self.zx.append(zx)
            self.zy.append(zy)
            self.nhits.append(nhits)

            E, vx, vy, vz = truth_map[evt]
            ELep, EtaLep = primaries_map[evt]
            self.targets.append((E, vx, vy, vz, ELep, EtaLep))

    def __len__(self):
        return len(self.zx)

    def __getitem__(self, idx):
        E, vx, vy, vz, ELep, EtaLep = self.targets[idx]
        # E, vx, vy, vz= self.targets[idx]
        return (
            self.zx[idx],
            self.zy[idx],
        ), {
            "E_nu": np.log10(E/1000), # Mev -> GeV
            "weight": self.sample_weights[idx],
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "E_lep": np.log10(ELep/1000), # MeV -> GeV
            "Eta_lep": EtaLep
        }


if __name__ == "__main__":

    input_files = glob.glob("/home/benwilson/data/pinpointG4_data/root/10000/*.root")

    num_bins = 2048
    num_layers = 100
    
    for fpath in input_files:

        output_torch_path = os.path.basename(fpath).replace(".root", ".pt")

        dataset = CNNProjectionDataset(
            fpath, bins=(num_bins, num_bins, num_layers)
        )

        logging.info(f"Saving dataset to {output_torch_path}")
        torch.save(dataset, output_torch_path)

        # break