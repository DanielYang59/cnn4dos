"""Perform eDOS occlusion experiments."""

import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Modify sys.path for shared components
root_dir = Path(__file__).parent
shared_components_dir = root_dir / "../shared_components/src"
sys.path.append(str(shared_components_dir.resolve()))

# TODO: need a fix for ruff warning than suppressing
from cnnPredictor import CNNPredictor  # noqa: E402
from dataLoader import DataLoader  # noqa: E402
from dosProcessor import DOSProcessor  # noqa: E402
from src.occlusionGenerator import occlusionGenerator  # noqa: E402
from src.occlusionPlotter import OcclusionPlotter  # noqa: E402
from src.utilities import get_fermi_level  # noqa: E402


def main():
    """Main function to execute occlusion experiments."""

    # Step 1: Load config and adsorbate DOS
    data_loader = DataLoader()
    config = data_loader.load_config(root_dir / "config.yaml")

    adsorbate_dos = data_loader.load_and_preprocess_adsorbate_dos(
        root_dir / config["path"]["adsorbate_dos_array_path"],
        config["occlusion"]["max_adsorbate_channels"],
    )

    # Step 2: Load original eDOS,
    # in shape (numSamplings, numOrbitals, 1)
    # and remove ghost state
    unshifted_dos = data_loader.load_unshifted_dos(
        Path(os.getcwd()) / config["occlusion"]["dos_array_name"]
    )
    dos_processor = DOSProcessor(unshifted_dos)
    processed_dos = dos_processor.remove_ghost_state()

    # Step 3: Generate occlusion arrays
    print("Generating occlusion arrays...")
    generator = occlusionGenerator(
        dos_array=processed_dos,
        occlusion_width=config["occlusion"]["occlusion_width"],
        occlusion_step=config["occlusion"]["occlusion_step"],
        dos_calculation_resolution=config["occlusion"][
            "dos_calculation_resolution"
        ],  # noqa: E501
    )
    occlusion_arrays = generator.generate_occlusion_arrays()
    print("Occlusion arrays generated.")

    # Step 4: Predict with CNN model
    # Load the CNN model
    cnn_model = tf.keras.models.load_model(
        root_dir / Path(config["path"]["cnn_model_path"])
    )
    # Create an instance of CNNPredictor
    cnn_predictor = CNNPredictor(loaded_model=cnn_model)

    # Calculate reference point
    ref_prediction = cnn_predictor.predict(processed_dos, adsorbate_dos)

    # Make prediction along each orbital
    predictions = []

    # Get dimensions
    num_occlusions = occlusion_arrays.shape[0]  # Number of occlusions
    numOrbitals = occlusion_arrays.shape[1]  # Number of orbitals

    # Initialize tqdm progress bar
    with tqdm(total=num_occlusions * numOrbitals, desc="Making Predictions") as pbar:
        for i in range(num_occlusions):  # Loop over the number of occlusions
            for j in range(numOrbitals):  # Loop over each orbital
                occluded_array = np.expand_dims(
                    occlusion_arrays[i, j], axis=-1
                )  # Reshape array
                prediction = cnn_predictor.predict(
                    occluded_array, adsorbate_dos
                )  # Make prediction
                predictions.append(prediction)  # Append to predictions list
                pbar.update(1)  # Update the progress bar

    # Reshape the predictions array to (num_occlusions, numOrbitals)
    predictions = np.array(predictions).reshape(num_occlusions, numOrbitals)

    # Subtract ref_prediction from each prediction
    predictions = predictions - ref_prediction

    # (Optional) Save predictions locally
    if config["occlusion"]["save_predictions"]:
        np.save(Path(os.getcwd()) / "occlusion_predictions.npy", predictions)

    # # (Optional) Load cached local predictions
    # predictions = np.load(
    #     Path(os.getcwd()) / "occlusion_predictions.npy"
    # )

    # Step 5: Read fermi level and plot occlusion
    fermi_level = get_fermi_level(
        working_dir=os.getcwd(),
        fermi_level_source=root_dir
        / Path(config["path"]["fermi_level_source"]),  # noqa: E501
    )

    plotter = OcclusionPlotter(predictions, config, fermi_level)
    plotter.plot_heatmap(orbitals=config["plotting"]["heatmap_orbitals"])
    # plotter.plot_line(orbitals=config['plotting']['line_orbitals'])


if __name__ == "__main__":
    main()