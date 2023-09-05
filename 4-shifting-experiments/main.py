#!/bin/usr/python3
# -*- coding: utf-8 -*-


import numpy as np
from pathlib import Path

from src.get_folders_in_dir import get_folders_in_dir
from src.plot_shifting import plot_shifting
from src.read_config import read_config
from src.ShiftingExperiment import ShiftingExperiment


def main(configfile="config.yaml"):
    # Read shifting experiment configs
    config = read_config(Path(configfile))


    # Read folders
    folders = get_folders_in_dir(
        directory_path=config["path"]["working_dir"],
        filter_file=config["shifting"]["dos_array_name"],
        )


    for d in folders:
        # Initiate shifting experiment class
        experiment = ShiftingExperiment(
            model_path=Path(config["path"]["cnn_model_path"]),
            adsorbate_path=Path(config["path"]["adsorbate_dos_array_path"]),
            array=np.load(d / config["shifting"]["dos_array_name"]),
            working_dir=Path(d),
        )


        # Perform shifting experiment
        shifted_arrays = experiment.generate_shifted_arrays(
            shift_range=config["shifting"]["shifting_range"],
            shift_step=config["shifting"]["shifting_step_length"],
            orbital_indexes=config["shifting"]["shifting_orbitals"],
            )

        differences = experiment.run_experiment(shifted_arrays)


        # Save shifting experiment results
        experiment.save_shifted_arrays(
            shifted_arrays=differences,
            filename=Path(d) / "shifting_diff.npy",
            )


    # Plot shifting experiment heatmap
    plot_shifting()


if __name__ == "__main__":
    main()
