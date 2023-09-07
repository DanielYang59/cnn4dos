#!/bin/usr/python3
# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
import warnings

class ShiftGenerator:
    def __init__(self, dos_array: np.ndarray, shifting_range: list, shifting_step: float, shifting_orbitals: list, dos_calculation_resolution: float, save_arrays: bool = False, save_path: Path = None):
        """
        Initialize the ShiftGenerator.

        Args:
            dos_array (np.ndarray): The pre-processed DOS array of shape (numSamplings, numOrbitals, 1).
            shifting_range (list): A list containing the start and end of the shifting range.
            shifting_step (float): The step length for each shift.
            shifting_orbitals (list): A list containing the indices of the orbitals to be shifted.
            dos_calculation_resolution (float): The energy resolution along the numSamplings axis of the DOS array.
            save_arrays (bool, optional): Whether to save the shifted arrays to disk. Defaults to False.
            save_path (Path, optional): The directory where to save the shifted arrays if save_arrays is True. Defaults to None.

        Raises:
            ValueError: If the shape of dos_array is not as expected, or if shifting parameters are not valid.
        """
        # Check the shape of dos_array
        numSamplings, numOrbitals, numChannels = dos_array.shape
        if numSamplings <= 500:
            warnings.warn("The number of samplings is not greater than 500.")
        if numOrbitals not in {1, 4, 9, 16}:
            raise ValueError("numOrbitals must be one of {1, 4, 9, 16}")
        if numChannels != 1:
            raise ValueError("numChannels must be 1")

        # Check shifting parameters
        start, end = shifting_range
        if start >= end:
            raise ValueError("The start of the shifting range should be smaller than the end.")
        if shifting_step <= 0:
            raise ValueError("shift_step should be greater than zero.")
        if not self._is_multiple_of(number=(end - start), base=shifting_step):
            raise ValueError("The shifting range (end - start) should be a multiple of shift_step.")
        if len(set(shifting_orbitals)) != len(shifting_orbitals):
            raise ValueError("Duplicate orbital indices found in shifting_orbitals.")
        if any((idx < 0 or idx >= numOrbitals) for idx in shifting_orbitals):
            raise ValueError(f"Invalid orbital indices in shifting_orbitals. Valid range is [0, {numOrbitals-1}]")
        if not self._is_multiple_of(shifting_step, dos_calculation_resolution):
            raise ValueError("shifting_step must be a multiple of dos_calculation_resolution.")

        self.dos_array = dos_array
        self.shifting_range = shifting_range
        self.shift_step = shifting_step
        self.shifting_orbitals = shifting_orbitals
        self.dos_calculation_resolution = dos_calculation_resolution
        self.save_arrays = save_arrays
        if save_arrays and save_path is None:
            raise ValueError("If save_arrays is True, save_path cannot be None.")
        self.save_path = save_path

    def _is_multiple_of(self, number, base):
        """
        Check if a given float number is a multiple of another float base, considering numerical inaccuracies.

        Args:
            number (float): The number to check.
            base (float): The base to divide by.

        Returns:
            bool: True if 'number' is a multiple of 'base', False otherwise.
        """
        quotient = number / base
        return np.isclose(quotient, round(quotient))

    def generate_shifted_arrays(self, folder_name: str):
        """
        Generate a series of shifted DOS arrays.

        Args:
        folder_name (str): The name of the folder for this particular dataset.

        Returns:
            list: A list of shifted DOS arrays.

        Side effect:
            Saves a single NumPy array containing all shifted arrays for this folder.
        """
        start, end = self.shifting_range
        shift_values = np.arange(start, end + self.shift_step, self.shift_step)
        shifted_arrays = []

        for shift_value in shift_values:
            # Create a copy to store the shifted array
            shifted_dos = np.copy(self.dos_array)

            # Calculate the number of indices to shift based on dos_calculation_resolution
            num_indices_to_shift = int(abs(shift_value) / self.dos_calculation_resolution)

            if num_indices_to_shift >= shifted_dos.shape[0]:
                raise ValueError(f"Skipping shift value {shift_value} as it is greater or equal to the data shape.")

            for orbital_idx in self.shifting_orbitals:
                # First, perform the shifting
                if np.isclose(shift_value, 0, atol=1e-9):  # shift_value ~ 0
                    shifted_data = shifted_dos[:, orbital_idx, :]
                elif shift_value > 0:
                    shifted_data = shifted_dos[:-num_indices_to_shift, orbital_idx, :]
                else:  # shift_value < 0
                    shifted_data = shifted_dos[num_indices_to_shift:, orbital_idx, :]

                # Next, perform the zero-padding
                padding = np.zeros((num_indices_to_shift, 1))
                if np.isclose(shift_value, 0, atol=1e-9):  # shift_value ~ 0
                    padded_data = shifted_data
                elif shift_value > 0:
                    padded_data = np.concatenate([shifted_data, padding])
                else: # shift_value < 0
                    padded_data = np.concatenate([padding, shifted_data])

                # Assign the padded data back to the array
                shifted_dos[:, orbital_idx, 0] = padded_data.flatten()
            shifted_arrays.append(shifted_dos)

        # Save generated shift arrays
        if self.save_arrays:
            # Create a directory for the specific settings if it doesn't exist
            folder_to_save = self.save_path / folder_name.name
            folder_to_save.mkdir(parents=True, exist_ok=True)

            # Save a single array combining all shifted arrays
            all_shifted_arrays = np.stack(shifted_arrays, axis=0)  # Assuming axis=0 stacks them as you want
            np.save(folder_to_save / "all_shifted_arrays.npy", all_shifted_arrays)

        return shifted_arrays
