#!/bin/usr/python3
# -*- coding: utf-8 -*-

from tensorflow.keras.models import load_model
import numpy as np
from pathlib import Path

class CNNPredictor:

    def __init__(self, model_dir: Path):
        """
        Initialize the CNNPredictor class and load the Keras CNN model.

        Args:
            model_dir (Path): The directory containing the saved Keras CNN model.

        Raises:
            FileNotFoundError: If the specified directory or model does not exist.
        """
        model_path = Path(model_dir) / "model"
        if not model_path.exists():
            raise FileNotFoundError(f"The specified model {model_path} does not exist.")
        self.model = load_model(model_path)

    def predict(self, dos_array: np.ndarray, adsorbate_dos_array: np.ndarray) -> np.ndarray:
        """
        Make predictions based on the DOS and adsorbate DOS arrays.

        Args:
            dos_array (np.ndarray): The processed DOS array of shape (numSamplings, numOrbitals, 1).
            adsorbate_dos_array (np.ndarray): The processed adsorbate DOS array of shape (numSamplings, numOrbitals, max_adsorbate_channels).

        Returns:
            np.ndarray: The prediction array.

        Raises:
            ValueError: If the shapes of the arrays are not as expected.
        """

        # Check shapes
        if dos_array.shape[-1] != 1:
            raise ValueError("The last dimension (numChannels) of DOS array must be 1.")

        if dos_array.shape[:-1] != adsorbate_dos_array.shape[:-1]:
            raise ValueError("The shapes of dos_array and adsorbate_dos_array must match in the first two dimensions.")

        # Append adsorbate DOS array to DOS array along the numChannels axis
        combined_array = np.concatenate([dos_array, adsorbate_dos_array], axis=-1)

        # Make predictions
        predictions = self.model.predict(combined_array)

        return predictions
