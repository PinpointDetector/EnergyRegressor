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

def create_projections(
    hits_df: pd.DataFrame,
    bins: tuple[int, int, int] = (128, 128, 150),
    layer_var: str = "hit_layerID",
    i=0
) -> tuple[int | None, int | None, np.ndarray | None, np.ndarray | None]:
    # if len(hits_df.query(f"{layer_var} >= 4 & {layer_var} <= 14")) == 0:
    if len(hits_df.query(f"{layer_var} >= 4")) == 0:
        return None, None, None, None

    x_half_size = bins[0] // 2
    y_half_size = bins[1] // 2

    mean_x, mean_y = (
        # hits_df.query(f"{layer_var} >= 4 & {layer_var} <= 14")[["pixel_x", "pixel_y"]]
        hits_df.query(f"{layer_var} >= 4")[["hit_colID", "hit_rowID"]]
        .mean()
        .astype(int)
        .values
    )

    x_min, x_max = mean_x - x_half_size, mean_x + x_half_size
    y_min, y_max = mean_y - y_half_size, mean_y + y_half_size

    # 148
    # 12788
    # 8596

    h_zx = hist.Hist(
        hist.axis.Variable(np.arange(-0.5, bins[2] + 0.5, 1)),  # layer axis
        hist.axis.Variable(np.arange(x_min - 0.5, x_max + 0.5, 1)),  # x pixel axis
        storage=hist.storage.Weight(),
    )

    h_zy = hist.Hist(
        hist.axis.Variable(np.arange(-0.5, bins[2] + 0.5, 1)),  # layer axis
        hist.axis.Variable(np.arange(y_min - 0.5, y_max + 0.5, 1)),  # y pixel axis
        storage=hist.storage.Weight(),
    )

    print("------")
    print("layer: ", ak.ravel(hits_df[layer_var]))
    print("col: ",   ak.ravel(hits_df["hit_rowID"]))
    print("row: ",   ak.ravel(hits_df["hit_rowID"]))
    print("------")

    h_zx.fill(ak.ravel(hits_df[layer_var]), ak.ravel(hits_df["hit_colID"]))
    h_zy.fill(ak.ravel(hits_df[layer_var]), ak.ravel(hits_df["hit_rowID"]))

    zx_projection = h_zx.values()
    zy_projection = h_zy.values()

    fig, ax = plt.subplots(ncols=2, figsize=(18,6))
    mplhep.hist2dplot(h_zx, ax=ax[0])
    mplhep.hist2dplot(h_zy, ax=ax[1])
    plt.savefig(f"{i}.png", dpi=300)
    


    return mean_x, mean_y, zx_projection, zy_projection



class CNNProjectionDataset(Dataset):
    """Dataset that stores binned pixel counts per event"""

    def __init__(
        self,
        file_name: str,
        bins: tuple[int, int, int],
    ):
        self.projections = []
        self.labels = []
        self.e_nu = []
        self.e_lepton = []
        self.vx = []
        self.vy = []
        self.vz = []

        root_file = uproot.open(file_name)
        hits_df = root_file["Hits/pixelHits"].arrays(library="pd")
        truth_df = root_file["event"].arrays(library="pd")

        print(hits_df.keys())

        event_ids = hits_df["event_id"].unique()
        logging.info(f"Processing {len(event_ids)} events")

        for i, event_id in tqdm(enumerate(event_ids)):

            if i > 3: break

            event_hits = hits_df.loc[hits_df["event_id"] == event_id]
            event_truth = truth_df[truth_df["evtID"] == event_id].iloc[0]

            mean_x, mean_y, zx_proj, zy_proj = create_projections(event_hits, bins,i=i)
            if (
                mean_x is None
                or mean_y is None
                or zx_proj is None
                or zy_proj is None
            ):
                # FIXME: Because of this the truth dataframe has more entries
                continue

            zx_tensor = torch.FloatTensor(zx_proj).unsqueeze(0)
            zx_tensor = torch.log(zx_tensor + 1)

            zy_tensor = torch.FloatTensor(zy_proj).unsqueeze(0)
            zy_tensor = torch.log(zy_tensor + 1)

            self.projections.append((zx_tensor, zy_tensor))

            self.e_nu.append(np.float32(event_truth["initE"]))
            vx = np.float32(event_truth["initX"])
            vy = np.float32(event_truth["initY"])
            vz = np.float32(event_truth["initZ"])
            self.vx.append(vx)
            self.vy.append(vy)
            self.vz.append(vz)

        logging.info(f"Created dataset with {len(self.projections)} samples")

    def __len__(self):
        return len(self.projections)

    def __getitem__(self, idx):
        zx_tensor, zy_tensor = self.projections[idx]
        targets = {
            "E_nu": np.float32(self.e_nu[idx]),
            "E_lepton": np.float32(self.e_lepton[idx]),
            "vx": np.float32(self.vx[idx]),
            "vy": np.float32(self.vy[idx]),
            "vy": np.float32(self.vz[idx]),
        }

        return (zx_tensor, zy_tensor), targets


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

        break