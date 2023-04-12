#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import yaml
from lib.dos_loader import dosLoader
from lib.list_projects import list_projects


if __name__ == "__main__":
    # Load configs
    with open("config.yaml") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    perturbation_analysis_dir = Path(cfg["path"]["perturbation_analysis_dir"])
    dos_filename = cfg["path"]["dos_filename"]
    adsorbate_dosfile = Path(cfg["path"]["adsorbate_dosfile"])
    label_dir = Path(cfg["path"]["label_dir"])

    adsorbate_maxAtoms = cfg["species"]["adsorbate_maxAtoms"]


    # User choose project
    project = list_projects(perturbation_analysis_dir / "data")


    # User input atom index
    atom_index = int(input("Index of centre atom (starts from one)?"))


    # Load perturbed data and append adsorbate DOS
    data_loader = dosLoader(
        dos_path= perturbation_analysis_dir / "data" / project / "1-perturbed",
        dos_filename=dos_filename.replace("INDEX", atom_index),
        append_adsorbate=True,
        adsorbate_dosfile=adsorbate_dosfile,
        adsorbate_numAtoms=adsorbate_maxAtoms,
        )

    loaded_dos = data_loader.loaded_dos


    # Load labels



    # Load CNN model


    # Make predictions with CNN model


    # Analyze prediction performance

