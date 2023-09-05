#!/bin/usr/python3
# -*- coding: utf-8 -*-


import numpy as np
from pathlib import Path


class OrbitalShifter:
    """
    A class to manage the shifting of specific orbitals in a 2D array.

    Attributes:
        array (np.ndarray): The array containing the orbital data.
        working_dir (str): The directory where the DOS file is located.
        dos_filename (str): The name of the DOS file.
    """

    def __init__(self, array, working_dir, dos_filename):
        """
        Initializes the OrbitalShifter class.

        Args:
            array (np.ndarray): The 2D array to be shifted.
            working_dir (str): The directory where the DOS file is located.
            dos_filename (str): The name of the DOS file.
        """
        self.array = array
        self.working_dir = working_dir
        self.dos_filename = dos_filename
        self.validate_file()


    def validate_file(self):
        """Validates if the DOS file exists and is numpy array in the given directory."""
        if not Path(self.working_dir, self.dos_filename).exists():
            raise FileNotFoundError(f"The DOS file {self.dos_filename} does not exist under {self.working_dir}.")


        # Check if DOS array is a numpy array
        try:
            loaded_array = np.load(Path(self.working_dir, self.dos_filename))
        except ValueError:
            raise ValueError("The DOS file could not be loaded as a NumPy array.")

        if not isinstance(loaded_array, np.ndarray):
            raise ValueError("The loaded DOS file is not a NumPy array.")


    def validate_parameters(self, shift_range, shift_step, orbital_indexes):
        """
        Validates the parameters for the shift operation.

        Args:
            shift_range (tuple): The range for the shift.
            shift_step (float): The step size for the shift.
            orbital_indexes (list): The list of orbital indexes to be shifted.

        Raises:
            ValueError: If any of the validation checks fail.
        """
        if shift_step <= 0:
            raise ValueError("Shifting step must be greater than zero.")

        if shift_range[0] >= shift_range[1]:
            raise ValueError("The lower value of the shifting range must be less than the upper value.")

        if (shift_range[1] - shift_range[0]) % shift_step != 0:
            raise ValueError("Shift range must be a multiple of shift step.")

        if len(orbital_indexes) != len(set(orbital_indexes)):
            raise ValueError("Duplicate orbital indexes are not allowed.")

        if not all(0 <= idx < 9 for idx in orbital_indexes):
            raise ValueError("All orbital indexes must be in range(9)")


    def shift_orbitals(self, shift_range, shift_step, orbital_indexes):
        """
        Performs the shift operation on the selected orbitals in the array.

        Args:
            shift_range (tuple): The range for the shift.
            shift_step (float): The step size for the shift.
            orbital_indexes (list): The list of orbital indexes to be shifted.

        Returns:
            list: A list of shifted arrays.
        """
        self.validate_parameters(shift_range, shift_step, orbital_indexes)

        # Generate shift values based on range and step
        shift_values = np.arange(shift_range[0], shift_range[1] + shift_step, shift_step)

        # Initialize list to store shifted arrays
        shifted_arrays = []

        # Loop through each shift value to generate shifted arrays
        for shift in shift_values:
            shifted_array = np.copy(self.array)
            for idx in orbital_indexes:
                if shift > 0:
                    shifted_array[:, idx] = np.pad(shifted_array[:, idx], (0, int(shift)), "constant")[:shifted_array.shape[0]]
                elif shift < 0:
                    shifted_array[:, idx] = np.pad(shifted_array[:, idx], (int(-shift), 0), "constant")[int(-shift):]
            shifted_arrays.append(shifted_array)

        return np.array(shifted_arrays)
