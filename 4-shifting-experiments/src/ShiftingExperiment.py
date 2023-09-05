#!/bin/usr/python3
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
from pathlib import Path


class ShiftingExperiment:
    def __init__(self, model_path: Path, adsorbate_path: Path, array: np.ndarray, remove_ghost_state: bool, max_adsorbate_channels: int, working_dir: Path = Path(".")):
        """
        Initialize the ShiftingExperiment class.

        Parameters:
            model_path (Path): Path to the saved CNN model.
            adsorbate_path (Path): Path to the saved adsorbate array.
            array (np.ndarray): The initial unshifted array.
            remove_ghost_state (bool): Flag to remove the ghost state by setting the first row to zero.
            max_adsorbate_channels (int): Maximum number of adsorbate channels for padding.
            working_dir (Path): The working directory for the experiment.
        """
        self.model = self.load_model(model_path)
        self.adsorbate_array = self.load_adsorbate_array(adsorbate_path)
        self.array = array
        self.remove_ghost_state = remove_ghost_state
        self.max_adsorbate_channels = max_adsorbate_channels
        self.working_dir = working_dir

        if self.remove_ghost_state:
            self.remove_ghost_states()


    def remove_ghost_states(self):
        """Set the first value of all orbitals in the DOS array to zero if remove_ghost_state is True."""
        self.array[0, :] = 0.0


    def load_model(self, model_path: Path):
        """Loads a CNN model from a given path."""
        if not model_path.exists():
            raise FileNotFoundError(f"The model file does not exist at {model_path}.")
        return tf.keras.models.load_model(str(model_path))


    def load_and_pad_adsorbate_array(self, adsorbate_path: Path):
        """Loads and pads the adsorbate array from a given path."""
        if not adsorbate_path.exists():
            raise FileNotFoundError(f"The adsorbate file does not exist at {adsorbate_path}.")

        array = np.load(adsorbate_path)

        shape = array.shape
        if shape[-1] > self.max_adsorbate_channels:
            raise ValueError(f"The array has more channels ({shape[-1]}) than max_adsorbate_channels ({self.max_adsorbate_channels}).")

        padding_size = self.max_adsorbate_channels - shape[-1]
        return np.pad(array, ((0, 0), (0, 0), (0, padding_size)), "constant")


    def generate_shifted_arrays(self, shift_range: tuple, shift_step: float, orbital_indexes: list) -> np.ndarray:
        """
        Generates shifted arrays based on the provided parameters.

        Parameters:
            shift_range (tuple): The range of shifts to apply.
            shift_step (float): The step size for shifts.
            orbital_indexes (list): List of orbital indexes to shift.

        Returns:
            np.ndarray: Array of shifted arrays.
        """
        # Validation checks
        if shift_step <= 0:
            raise ValueError("Shifting step must be greater than zero.")
        if shift_range[0] >= shift_range[1]:
            raise ValueError("The lower value of the shifting range must be less than the upper value.")
        if (shift_range[1] - shift_range[0]) % shift_step != 0:
            raise ValueError("Shift range must be a multiple of shift step.")
        if len(orbital_indexes) != len(set(orbital_indexes)):
            raise ValueError("Duplicate orbital indexes are not allowed.")
        if not all(0 <= idx < self.array.shape[1] for idx in orbital_indexes):
            raise ValueError(f"All orbital indexes must be in range(0, {self.array.shape[1]})")

        num_shifts = int((shift_range[1] - shift_range[0]) / shift_step) + 1
        shifted_arrays = np.zeros((num_shifts,) + self.array.shape)
        for i, shift in enumerate(np.arange(shift_range[0], shift_range[1] + shift_step, shift_step)):
            shifted_array = np.copy(self.array)
            for idx in orbital_indexes:
                shifted_array[:, idx] = np.roll(self.array[:, idx], shift)
                if shift > 0:
                    shifted_array[:shift, idx] = 0
                elif shift < 0:
                    shifted_array[shift:, idx] = 0
            shifted_arrays[i] = shifted_array

        return shifted_arrays


    def run_experiment(self, shifted_arrays: np.ndarray):
        """Runs the shifting experiment and returns the prediction differences."""
        differences = []
        original_array = shifted_arrays[0]  # Assuming the first one is the unshifted array

        original_input = self.preprocess_input(original_array)
        original_prediction = self.model.predict(original_input)

        for shifted_array in shifted_arrays:
            shifted_input = self.preprocess_input(shifted_array)
            shifted_prediction = self.model.predict(shifted_input)

            difference = np.abs(original_prediction - shifted_prediction)
            differences.append(difference)

        return np.array(differences)


    def preprocess_input(self, array: np.ndarray) -> np.ndarray:
        """Preprocesses an input array for model prediction."""
        reshaped_array = np.expand_dims(array, axis=-1)  # Reshape to (NEDOS, orbitals, 1)
        concatenated_array = np.concatenate([reshaped_array, self.adsorbate_array], axis=-1)
        return concatenated_array


    def save_shifted_arrays(self, shifted_arrays: np.ndarray, filename: str = "shifted_arrays.npy"):
        """
        Saves the generated shifted arrays to the working directory.

        Parameters:
            shifted_arrays (np.ndarray): The shifted arrays to save.
            filename (str): The name of the saved file.
        """
        save_path = self.working_dir / filename
        np.save(save_path, shifted_arrays)
