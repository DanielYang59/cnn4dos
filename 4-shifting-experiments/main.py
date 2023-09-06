#!/bin/usr/python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import tensorflow as tf

from src.cnnPredictor import CNNPredictor
from src.dataLoader import DataLoader
from src.dosProcessor import DOSProcessor
from src.shiftGenerator import ShiftGenerator
from src.shiftPlotter import shiftPlotter
from src.utilities import get_folders_in_dir

def main():
    # Step 1: Load config and adsorbate DOS
    data_loader = DataLoader()
    config = data_loader.load_config('path/to/config.yaml')

    adsorbate_dos = data_loader.load_and_preprocess_adsorbate_dos(config['path']['adsorbate_dos_array_path'])

    # Step 2: List all matched folders
    working_dir = config['path']['working_dir']
    folders = get_folders_in_dir(working_dir, filter_file=config['shifting']['dos_array_name'])

    # Load the CNN model
    cnn_model = tf.keras.models.load_model(Path(config['path']['cnn_model_path']) / "model")

    # Create an instance of CNNPredictor
    cnn_predictor = CNNPredictor(cnn_model)

    # Create an empty list to store predictions for future plotting
    all_predictions = []

    # Step 3: Loop through each folder
    for folder in folders:
        dos_file_path = folder / config['shifting']['dos_array_name']

        # a. Load DOS array with DOSProcessor
        dos_processor = DOSProcessor(dos_file_path)
        processed_dos = dos_processor.remove_ghost_state()

        # b. Generate shifting arrays with ShiftGenerator
        shift_gen = ShiftGenerator(processed_dos, config['shifting'])
        shifted_dos_arrays = shift_gen.generate_shifted_arrays()

        # c. Feed each shifted array into the CNN model for prediction
        predictions = []
        for shifted_dos in shifted_dos_arrays:
            prediction = cnn_predictor.predict(shifted_dos, adsorbate_dos)
            predictions.append(prediction)

        # d. Feed the unshifted DOS array into the CNN model for a reference point
        ref_prediction = cnn_predictor.predict(processed_dos, adsorbate_dos)

        # e. Store the predictions for future plotting
        all_predictions.append({"folder": folder, "predictions": np.array(predictions), "reference": ref_prediction})

    # # Step 4: Plot
    # shiftPlotter(all_predictions)


if __name__ == "__main__":
    main()
