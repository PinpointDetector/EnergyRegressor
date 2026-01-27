import glob
import os
import logging
import argparse
import torch
from omegaconf import DictConfig
from source.dataset import CNNProjectionDataset


def preprocess_data(cfg: DictConfig):

    input_files = glob.glob(cfg.preprocessing.input_dirpath)

    assert len(input_files) > 0, f"No input files found in {cfg.preprocessing.input_dirpath}"
    logging.info(f"Found {len(input_files)} input files to process.")

    output_dir = cfg.preprocessing.output_dirpath
    logging.info(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)


    for fpath in input_files:

        output_torch_path = os.path.basename(fpath).replace(".root", ".pt")
        output_torch_path = os.path.join(output_dir, output_torch_path)

        dataset = CNNProjectionDataset(
            fpath, bins=(cfg.preprocessing.num_bins_x, cfg.preprocessing.num_bins_y, cfg.preprocessing.num_bins_z)
        )

        logging.info(f"Saving dataset to {output_torch_path}")
        torch.save(dataset, output_torch_path)



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
