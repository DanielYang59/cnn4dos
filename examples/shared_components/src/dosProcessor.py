"""Utility script for eDOS preprocess: remove ghost states."""

import warnings

import numpy as np


class DOSProcessor:
    def __init__(self, dos_array: np.ndarray):
        """
        Initialize the DOSProcessor with a pre-loaded eDOS array.

        Args:
            dos_array (np.ndarray): A pre-loaded eDOS array with
                shape (numSamplings, numOrbitals, 1).

        Raises:
            TypeError: If the given eDOS array is not a numpy array.
            ValueError: If the shape of the given eDOS array is not as
                expected, or if numSamplings <= 500,
                or if numOrbitals not in {1, 4, 9, 16}.
        """
        if not isinstance(dos_array, np.ndarray):
            raise TypeError("The given eDOS array must be a numpy array.")

        numSamplings, numOrbitals, _ = dos_array.shape

        if numSamplings <= 500:
            warnings.warn("Number of samplings is not greater than 500.")

        if numOrbitals not in {1, 4, 9, 16}:
            raise ValueError("numOrbitals must be one of {1, 4, 9, 16}")

        self.dos_array = dos_array

    def remove_ghost_state(self, remove_ghost: bool = False) -> np.ndarray:
        """
        Optionally remove ghost states from the Density of States (DOS) array.

        Args:
            remove_ghost (bool, optional): If True, sets the first
                values (index 0) along the numOrbitals axis to 0.0.

        Returns:
            np.ndarray: eDOS array with ghost states removed, if specified.
                Shape remains (numSamplings, numOrbitals, 1).
        """
        if remove_ghost:
            self.dos_array[0, :] = 0.0

        return self.dos_array
