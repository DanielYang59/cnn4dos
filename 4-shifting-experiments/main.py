#!/bin/usr/python3
# -*- coding: utf-8 -*-


from pathlib import Path

from src.get_folders_in_dir import get_folders_in_dir
from src.plot_shifting import plot_shifting
from src.load_adsorbate import load_and_transpose_dos
from src.load_config import load_config
from src.load_cnn_model import load_cnn_model


def main(configfile="config.yaml"):
    # Load configs
    config = load_config(Path(configfile))


    # Read folders
    folders = get_folders_in_dir(
        directory_path=config["path"]["working_dir"],
        filter_file=config["shifting"]["dos_array_name"],
        )


    # Load and transpose adsorbate array
    adsorbate_dos_array = load_and_transpose_dos(Path(config["path"]["adsorbate_dos_array_path"]))


    # Load CNN model
    cnn_model = load_cnn_model(Path(config["path"]["cnn_model_path"]))
    cnn_model.summary()


    # G


    # # Plot shifting experiment as heatmap
    # plot_shifting(folders)


if __name__ == "__main__":
    main()
