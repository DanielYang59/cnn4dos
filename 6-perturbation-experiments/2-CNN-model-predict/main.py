#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
import yaml
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from lib.dos_loader import dosLoader
from lib.list_projects import list_projects


if __name__ == "__main__":
    # Load configs
    with open("config.yaml") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    adsorbate_dosfile = Path(cfg["path"]["adsorbate_dosfile"])
    cnn_model_dir = Path(cfg["path"]["cnn_model_dir"])
    dos_filename = cfg["path"]["dos_filename"]
    label_dir = Path(cfg["path"]["label_dir"])
    perturbation_analysis_dir = Path(cfg["path"]["perturbation_analysis_dir"])

    adsorbate_maxAtoms = cfg["species"]["adsorbate_maxAtoms"]


    # User choose project
    project = list_projects(perturbation_analysis_dir / "data")


    # User input atom index
    atom_index = int(input("Index of centre atom (starts from one)?"))


    # Load perturbed data and append adsorbate DOS
    data_loader = dosLoader(
        dos_path= perturbation_analysis_dir / "data" / project / "1-perturbed",
        dos_filename=dos_filename.replace("INDEX", str(atom_index)),
        append_adsorbate=True,
        adsorbate_dosfile=adsorbate_dosfile,
        adsorbate_numAtoms=adsorbate_maxAtoms,
        )

    loaded_dos = data_loader.loaded_dos


    # Load labels
    labels = data_loader.load_label(label_dir / f"{project}.csv")
    assert labels.keys() == loaded_dos.keys()


    # Load CNN model
    model = tf.keras.models.load_model(cnn_model_dir)


    # Make predictions with CNN model (no preprocessing)
    features = np.array(list(loaded_dos.values()))
    predictions = model.predict(features, verbose=0).flatten()


    # Analyze prediction performance
    mae = mean_absolute_error(
        y_true=list(labels.values()),
        y_pred=predictions)
    print(f"Prediction MAE is {mae:.4f} eV.")
