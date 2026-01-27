from omegaconf import DictConfig, OmegaConf
import hydra
from source.train import run_training
from source.preprocess import preprocess_data
from source.test import run_testing
import logging
# from source.test import run_testing

@hydra.main(config_path="config", config_name="config", version_base="1.2")
def my_app(cfg: DictConfig) -> None:

    # Set log level
    level = logging.DEBUG if cfg.logging.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


    if cfg.mode == "preprocess":
        preprocess_data(cfg)

    elif cfg.mode == "train":
        run_training(cfg)    
    
    if cfg.mode == "test":
        run_testing(cfg)

    else:
        logging.error(f"Unknown mode: {cfg.mode}")
    


if __name__ == "__main__":
    my_app()