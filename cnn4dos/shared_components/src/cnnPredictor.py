"""Infer with pre-trained CNN model."""

from pathlib import Path

import numpy as np
import tensorflow as tf


class CNNPredictor:
    def __init__(self, model_path=None, loaded_model=None):
        """
        Initialize the CNNPredictor class.

        Args:
            model_path (str, optional): The file path to the saved Keras model.
            loaded_model (tf.keras.Model, optional): An already loaded Keras model.

        Raises:
            ValueError: If both model_path and loaded_model are provided.
        """

        if model_path and loaded_model:
            raise ValueError(
                "You can either provide a model_path or a loaded_model, not both."
            )

        if model_path:
            self.model = tf.keras.models.load_model(Path(model_path) / "model")

        elif loaded_model:
            if not isinstance(loaded_model, tf.keras.Model):
                raise TypeError("loaded_model should be of type tf.keras.Model")
            self.model = loaded_model

        else:
            raise ValueError("Either model_path or loaded_model should be provided.")

    def predict(
        self, dos_array: np.ndarray, adsorbate_dos_array: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions based on the eDOS and adsorbate eDOS arrays.

        Args:
            dos_array (np.ndarray): The processed eDOS array of shape (numSamplings, numOrbitals, 1).
            adsorbate_dos_array (np.ndarray): The processed adsorbate eDOS array of shape (numSamplings, numOrbitals, max_adsorbate_channels).

        Returns:
            np.ndarray: The prediction array.

        Raises:
            ValueError: If the shapes of the arrays are not as expected.
        """

        # Check shapes
        if dos_array.shape[-1] != 1:
            raise ValueError(
                "The last dimension (numChannels) of eDOS array must be 1."
            )

        if dos_array.shape[:-1] != adsorbate_dos_array.shape[:-1]:
            raise ValueError(
                "The shapes of dos_array and adsorbate_dos_array must match in the first two dimensions."
            )

        # Append adsorbate eDOS array to eDOS array along the numChannels axis
        combined_array = np.concatenate([dos_array, adsorbate_dos_array], axis=-1)

        # Make predictions with CNN model
        predictions = self.model.predict(
            np.expand_dims(combined_array, axis=0), verbose=0
        ).flatten()

        return predictions
