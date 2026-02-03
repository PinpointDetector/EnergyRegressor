# Neutrino Regression Network

## Summary

This repository contains the framework for training the Pinpoint neutrino energy regression code

## Getting Started

This code is configured using [`hydra`](https://hydra.cc) and [`pytorch lightning`](https://lightning.ai/docs/pytorch/stable/)

TODO: Add an enviroment file

The code is steered by `nuRegressor.py` which can operate in in one of three modes:

- `preprocess`: Run the preprocessing step to convert the Pinpoint Geant4 root NTuples into the `.pt` pytorch files which contain the event image projrections and the training targets. An example command looks like:

```bash
python3 nuRegressor.py mode=preprocess preprocessing.input_dirpath=../path/to/Ntuples/10000/*.root preprocessing.output_dirpath=data/10000
```

- `train`: Run the training step. Unless overrridden, the output of the training and tensorboard logs will be saved to `logs/energy_regression_train/<date-time>/`. An example command looks like:
```bash
python3 nuRegressor.py mode=train training.learning_rate=0.00001

```

- `test`: Run inference on test dataset and make the resolution plots. Will load the `logs/energy_regression_train/<data-time>/.hydra/config.yaml` config to reload the checkpointed model:
```bash
python3 nuRegressor.py mode=test testing.run_dir=logs/energy_regression_train/2026-01-31_16-12-00/
```



The `config/config.yaml` file contains all the parameters that will be used for the preprocessing, training and testing. Through hydra you can override any of the parameters from the command line when you run `nuRegressor.py`. For example, if you just wanted to regress the neutrino and primary lepton energy during training you could run
```bash
python3 nuRegressor.py mode=train training.targets=["E_nu","E_lep"]
```