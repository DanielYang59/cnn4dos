#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import tensorflow as tf
import sys

# Modify sys.path for shared components
root_dir = Path(__file__).parent
shared_components_dir = root_dir / "../shared_components/src"
sys.path.append(str(shared_components_dir.resolve()))

from src.occlusionGenerator import occlusionGenerator
from src.occlusionPlotter import occlusionPlotter

from cnnPredictor import CNNPredictor
from dataLoader import DataLoader
from dosProcessor import DOSProcessor
from utilities import get_folders_in_dir

def main():
    """Main function to execute occlusion experiments."""

    # Step 1: Load config and adsorbate DOS
    data_loader = DataLoader()
    config = data_loader.load_config(root_dir / "config.yaml")

    adsorbate_dos = data_loader.load_and_preprocess_adsorbate_dos(
        root_dir / config['path']['adsorbate_dos_array_path'],
        config['occlusion']['max_adsorbate_channels']
    )

    # Step 2: Load original DOS in shape (numSamplings, numOrbitals, 1)
    original_DOS = data_loader.load_unshifted_dos(Path(os.getcwd()) / config['occlusion']['dos_array_name'])

    # Step 3: Generate occlusion arrays
    generator = occlusionGenerator()
    occlusion_arrays = generator.generate_occlusion_arrays()

    # # Step 4: Predict with CNN model
    # # Load the CNN model
    # cnn_model = tf.keras.models.load_model(Path(config['path']['cnn_model_path']))
    # # Create an instance of CNNPredictor
    # cnn_predictor = CNNPredictor(loaded_model=cnn_model)

    # # Step 5: Plot
    # plotter = ShiftPlotter(all_predictions, config)
    # plotter.plot(save_dir=f"figures{os.sep}{str(working_dir).split(os.sep)[-1]}")

if __name__ == "__main__":
    main()