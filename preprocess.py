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

        self.zx = []
        self.zy = []
        self.nhits = []
        self.targets = []

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

            # nhits = torch.from_numpy(np.log1p(nhits)).unsqueeze(0).float()

            self.zx.append(zx)
            self.zy.append(zy)
            self.nhits.append(nhits)

            E, vx, vy, vz = truth_map[evt]
            self.targets.append((E, vx, vy, vz))

    def __len__(self):
        return len(self.zx)

    def __getitem__(self, idx):
        E, vx, vy, vz = self.targets[idx]
        return (
            self.zx[idx],
            self.zy[idx],
        ), {
            "E_nu": np.log10(E/1000), # Mev -> GeV
            "vx": vx,
            "vy": vy,
            "vz": vz,
        }


if __name__ == "__main__":

    input_files = glob.glob("/home/benwilson/data/pinpointG4_data/root/10000/*.root")

    num_bins = 1048
    num_layers = 100
    
    for fpath in input_files:

        output_torch_path = os.path.basename(fpath).replace(".root", ".pt")

        dataset = CNNProjectionDataset(
            fpath, bins=(num_bins, num_bins, num_layers)
        )

        logging.info(f"Saving dataset to {output_torch_path}")
        torch.save(dataset, output_torch_path)

        # break