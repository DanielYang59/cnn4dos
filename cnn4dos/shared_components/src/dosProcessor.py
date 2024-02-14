#!/bin/usr/python3
# -*- coding: utf-8 -*-

import warnings

import numpy as np


class DOSProcessor:
    def __init__(self, dos_array: np.ndarray):
        """
        Initialize the DOSProcessor with a pre-loaded DOS array.

        Args:
            dos_array (np.ndarray): A pre-loaded DOS array with shape (numSamplings, numOrbitals, 1).

        Raises:
            TypeError: If the given DOS array is not a numpy array.
            ValueError: If the shape of the given DOS array is not as expected, or if numSamplings <= 500, or if numOrbitals not in {1, 4, 9, 16}.
        """
        if not isinstance(dos_array, np.ndarray):
            raise TypeError("The given DOS array must be a numpy array.")

        numSamplings, numOrbitals, _ = dos_array.shape

        if numSamplings <= 500:
            warnings.warn(
                "The number of samplings is not greater than 500, please double-check if DOS is in correct shape."
            )

        if numOrbitals not in {1, 4, 9, 16}:
            raise ValueError("numOrbitals must be one of {1, 4, 9, 16}")

        self.dos_array = dos_array

    def remove_ghost_state(self, remove_ghost: bool = False) -> np.ndarray:
        """
        Optionally remove ghost states from the Density of States (DOS) array.

        Args:
            remove_ghost (bool, optional): If True, sets the first values (index 0) along the numOrbitals axis to 0.0.

        Returns:
            np.ndarray: DOS array with ghost states removed, if specified. Shape remains (numSamplings, numOrbitals, 1).
        """
        if remove_ghost:
            self.dos_array[0, :] = 0.0

        return self.dos_array
