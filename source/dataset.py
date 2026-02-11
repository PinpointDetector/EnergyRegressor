
import logging
import random
import torch
import uproot
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np


def compute_energy_weights(energies, n_bins=100):
    logE = np.log10(energies)

    bins = np.linspace(logE.min(), logE.max(), n_bins + 1)
    bin_idx = np.digitize(logE, bins) - 1

    counts = np.bincount(bin_idx, minlength=n_bins)
    counts[counts == 0] = 1  # avoid div by zero

    weights = 1.0 / counts[bin_idx]
    weights /= weights.mean()  # normalize

    return weights


def create_projections(hit_layer, hit_col, hit_row, bins):

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

    xy, _, _ = np.histogram2d(
        hit_col,
        hit_row,
        bins=(bins[0], bins[1]),
        range=((mean_x - x_half, mean_x + x_half), (mean_y - y_half, mean_y + y_half)),
    )

    return int(mean_x), int(mean_y), zx, zy, xy


def get_scint_hits_img(scint_layer, scint_col, scint_edep, nlayers, nbars=27): # nbarsX = 27, nbarsY = 20

    det = np.zeros((nlayers, nbars))

    for layer, col, edep in zip(scint_layer, scint_col, scint_edep):
        det[layer, col] += edep

    # det = det.unsqueeze(0)  # (1, 74, 26)

    return det


class CNNProjectionDataset(Dataset):
    def __init__(self, file_name, bins):
        root_file = uproot.open(file_name)

        truth = root_file["event"].arrays(library="np")
        primaries = root_file["primaries"].arrays(library="np")
        geom = root_file["geometry"].arrays(library="np")
        hits = root_file["Hits/pixelHits"]
        scints = root_file["Hits/scintHits"]

        nlayers = geom["nLayers"][0]

        event_id   = hits["event_id"].array(library="np")
        hit_layer  = hits["hit_layerID"].array(library="np")
        hit_col    = hits["hit_colID"].array(library="np")
        hit_row    = hits["hit_rowID"].array(library="np")

        scint_layer  = scints["layerID"].array(library="np")
        scint_col    = scints["colID"].array(library="np")
        scint_row    = scints["rowID"].array(library="np")
        scint_edep   = scints["edep"].array(library="np")

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
        neutrino_mask = np.isin(np.abs(primaries["PDG"]), [12, 14, 16])
        charged_hadron_mask = np.isin(np.abs(primaries["charge"]), [1])

        primary_leptons = {
            key: value[lepton_mask]
            for key, value in primaries.items()
        }

        primary_charged_hadrons = {
            key: value[~lepton_mask & ~neutrino_mask & charged_hadron_mask]
            for key, value in primaries.items()
        }

        primary_lepton_map = {
            evt: (E, theta)
                for evt, E, theta in zip(
                    primary_leptons["evtID"],
                    primary_leptons["E"],
                    primary_leptons["Eta"]
                )
        }

        primary_charged_hadrons_map = {
            evt: (E, theta)
                for evt, E, theta in zip(
                    primary_charged_hadrons["evtID"],
                    primary_charged_hadrons["E"],
                    primary_charged_hadrons["Eta"]
                )     
        }

        self.zx = []
        self.zy = []
        self.xy = []
        self.scint_zx = []
        self.scint_zy = []
        self.nhits = []
        self.tot_scint_edep = []
        self.targets = []
        self.sample_weights = compute_energy_weights(truth["initE"])

        # Group hits ONCE
        unique_evt, evt_index = np.unique(event_id, return_inverse=True)

        for i, evt in tqdm(enumerate(unique_evt), total=len(unique_evt)):
            idx = np.where(evt_index == i)[0]

            if evt not in truth_map:
                continue

            mean_x, mean_y, zx, zy, xy = create_projections(
                np.ravel(hit_layer[idx][0]),
                np.ravel(hit_col[idx][0]),
                np.ravel(hit_row[idx][0]),
                bins,
            )
            if zx is None:
                continue
            nhits = len(np.ravel(hit_row[idx][0]))

            scint_zx = get_scint_hits_img(
                np.ravel(scint_layer[idx][0]),
                np.ravel(scint_col[idx][0]),
                np.ravel(scint_edep[idx][0]),
                nlayers,
                nbars=27
                )
            
            scint_zy = get_scint_hits_img(
                np.ravel(scint_layer[idx][0]),
                np.ravel(scint_row[idx][0]),
                np.ravel(scint_edep[idx][0]),
                nlayers,
                nbars=20
                )

            zx = torch.from_numpy(np.log1p(zx)).unsqueeze(0).float()
            zy = torch.from_numpy(np.log1p(zy)).unsqueeze(0).float()
            xy = torch.from_numpy(np.log1p(xy)).unsqueeze(0).float()

            scint_zx = torch.from_numpy(np.log1p(scint_zx)).unsqueeze(0).float()
            scint_zy = torch.from_numpy(np.log1p(scint_zy)).unsqueeze(0).float()

            nhits = torch.from_numpy(np.log1p([nhits])).unsqueeze(0).float()
            tot_scint_edep = np.sum(np.ravel(scint_edep[idx][0]))

            self.zx.append(zx)
            self.zy.append(zy)
            self.xy.append(xy)
            self.scint_zx.append(scint_zx)
            self.scint_zy.append(scint_zy)
            self.nhits.append(nhits)
            self.tot_scint_edep.append(tot_scint_edep)

            E, vx, vy, vz = truth_map[evt]
            try:
                ELep, EtaLep = primary_lepton_map[evt]
            except KeyError: # NC events have no primary lepton, fill with dummy values
                ELep, EtaLep = 1e-6, 1e-6 # small value to avoid issues with log scale
            
            evt_mask = primary_charged_hadrons["evtID"] == evt
            sum_EHad = primary_charged_hadrons["E"][evt_mask].sum() if len(primary_charged_hadrons["E"][evt_mask])>0 else 1e-6

            self.targets.append((E, vx, vy, vz, ELep, EtaLep, sum_EHad))

    def __len__(self):
        return len(self.zx)

    def __getitem__(self, idx):
        E, vx, vy, vz, ELep, EtaLep, EHad = self.targets[idx]
        # E, vx, vy, vz= self.targets[idx]
        return (
            self.zx[idx],
            self.zy[idx],
            self.scint_zx[idx],
            self.scint_zy[idx],
            # [self.nhits[idx], self.tot_scint_edep[idx]]
            # self.xy[idx],
        ), {
            "E_nu": np.log10(E/1000), # Mev -> GeV
            "weight": self.sample_weights[idx],
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "E_lep": np.log10(ELep/1000), # MeV -> GeV
            "Eta_lep": EtaLep,
            "E_had": np.log10(EHad/1000) # MeV -> GeV
        }


class CombinedDataset(Dataset):
    def __init__(self, pt_files):
        self.datasets = [torch.load(f, map_location="cpu", weights_only=False, mmap=True) for f in pt_files]
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
        train_test_val_split=(0.7, 0.2, 0.1),
    ):
        super().__init__()
        self.pt_files = pt_files
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_test_val_split[0]
        self.test_split = train_test_val_split[1]
        self.val_split = train_test_val_split[2]

        assert abs(self.train_split + self.test_split + self.val_split - 1.0) < 1e-6, "Splits must sum to 1.0"

        self.setup()

    def setup(self, stage=None):
        files = list(self.pt_files)

        rng = random.Random(42)
        rng.shuffle(files)

        n = len(files)
        n_test = int(0.2 * n)
        n_val = int(0.1 * n)

        test_files = files[:n_test]
        val_files = files[n_test:n_test + n_val]
        train_files = files[n_test + n_val:]

        self.train_ds = CombinedDataset(train_files)
        self.val_ds   = CombinedDataset(val_files)
        self.test_ds  = CombinedDataset(test_files)

        logging.info(f"Train events: {len(self.train_ds)}")
        logging.info(f"Val events:   {len(self.val_ds)}")
        logging.info(f"Test events:  {len(self.test_ds)}")

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