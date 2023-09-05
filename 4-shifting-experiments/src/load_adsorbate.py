#!/bin/usr/python3
# -*- coding: utf-8 -*-


import numpy as np
from pathlib import Path


def load_and_transpose_dos(filepath):
    """
    Load an adsorbate DOS array from a .npy file and transpose its shape.

    Parameters:
        filepath (str): The path to the .npy file containing the DOS array.

    Returns:
        np.ndarray: The transposed adsorbate DOS array.

    Raises:
        ValueError: If the number of orbitals is not in {1, 4, 9, 16}.
        FileNotFoundError: If the file doesn't exist.
    """

    # Check if the file exists
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"The file {filepath} does not exist.")

    # Load the array from the .npy file
    dos_array = np.load(file_path)

    # Validate shape of the loaded array
    if len(dos_array.shape) != 3:
        raise ValueError("The loaded array must have three dimensions: (numChannels, numSamplings, numOrbitals).")

    # Extract dimensions
    numChannels, numSamplings, numOrbitals = dos_array.shape

    # Check if numOrbitals is in the valid set
    if numOrbitals not in {1, 4, 9, 16}:
        raise ValueError("The number of orbitals must be one of {1, 4, 9, 16}.")

    # Transpose the array: (numChannels, numSamplings, numOrbitals) -> (numSamplings, numOrbitals, numChannels)
    transposed_dos_array = np.transpose(dos_array, (1, 2, 0))

    return transposed_dos_array
