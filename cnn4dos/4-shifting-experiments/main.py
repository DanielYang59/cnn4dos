"""eDOS shifting experiment main."""

import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from src.shiftGenerator import ShiftGenerator
from src.shiftPlotter import ShiftPlotter
from src.utilities import get_folders_in_dir
from tqdm import tqdm

sys.path.append("../shared_components/src")
from cnnPredictor import CNNPredictor
from dataLoader import DataLoader
from dosProcessor import DOSProcessor


def main():
    # Step 1: Load config and adsorbate DOS
    data_loader = DataLoader()
    config = data_loader.load_config("config.yaml")

    adsorbate_dos = data_loader.load_and_preprocess_adsorbate_dos(
        config["path"]["adsorbate_dos_array_path"],
        config["shifting"]["max_adsorbate_channels"],
    )

    # Step 2: List all matched folders
    working_dir = Path(config["path"]["working_dir"])
    folders = get_folders_in_dir(
        working_dir, filter_file=config["shifting"]["dos_array_name"]
    )

    # Load the CNN model
    cnn_model = tf.keras.models.load_model(Path(config["path"]["cnn_model_path"]))

    # Create an instance of CNNPredictor
    cnn_predictor = CNNPredictor(loaded_model=cnn_model)

    # Create an empty list to store predictions for future plotting
    all_predictions = {}
    prediction_save_base = Path(config["path"]["prediction_saving_path"])

    # Step 3: Loop through each folder
    for i, folder in enumerate(tqdm(folders, desc="Processing folders")):
        print(f"Processing folder {folder.name} ({i + 1} out of {len(folders)})...")
        dos_file_path = folder / config["shifting"]["dos_array_name"]

        # a. Load DOS array with DOSProcessor
        unshifted_dos = data_loader.load_unshifted_dos(dos_file_path)
        dos_processor = DOSProcessor(unshifted_dos)
        processed_dos = dos_processor.remove_ghost_state()

        # b. Generate shifting arrays with ShiftGenerator
        shift_gen = ShiftGenerator(
            processed_dos,
            dos_calculation_resolution=config["shifting"]["dos_calculation_resolution"],
            shifting_range=config["shifting"]["shifting_range"],
            shifting_step=config["shifting"]["shifting_step"],
            shifting_orbitals=config["shifting"]["shifting_orbitals"],
        )
        shifted_dos_arrays = shift_gen.generate_shifted_arrays()

        # c. Feed each shifted array into the CNN model for prediction
        predictions = []
        for shifted_dos in shifted_dos_arrays:
            prediction = cnn_predictor.predict(shifted_dos, adsorbate_dos)
            predictions.append(prediction)

        # d. Feed the unshifted DOS array into the CNN model for a reference point
        ref_prediction = cnn_predictor.predict(processed_dos, adsorbate_dos)

        # Subtract ref_prediction from each prediction
        predictions = np.array(predictions) - ref_prediction

        # e. Save the predictions for future plotting
        all_predictions[folder.name] = predictions

        # Extract the last part of the working directory
        last_part_of_working_dir = Path(working_dir).name

        # Create a specific save path for each folder's predictions
        prediction_save_path = (
            prediction_save_base
            / last_part_of_working_dir
            / f"shifting_range_{config['shifting']['shifting_range']}-shifting_step_{config['shifting']['shifting_step']}-orbital_{config['shifting']['shifting_orbitals']}"
            / folder.name
        )
        prediction_save_path.mkdir(parents=True, exist_ok=True)

        # Save the predictions to disk
        np.save(prediction_save_path / "predictions.npy", predictions)

    # Step 4: Plot
    plotter = ShiftPlotter(all_predictions, config)
    plotter.plot(save_dir=f"figures{os.sep}{str(working_dir).split(os.sep)[-1]}")


if __name__ == "__main__":
    main()
