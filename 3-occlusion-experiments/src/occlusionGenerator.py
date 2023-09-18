#!/bin/usr/python3
# -*- coding: utf-8 -*-

import numpy as np
import warnings

class occlusionGenerator:
    """
    Class for generating Density of States (DOS) arrays with occlusion.
    """

    def __init__(self, dos_array: np.ndarray, occlusion_width: int, occlusion_step: int,  dos_calculation_resolution: float):
        """
        Initialize the OcclusionGenerator.

        Args:
            dos_array (np.ndarray): The original Density of States (DOS) array.
            occlusion_width (int): The width of the occlusion window. Actual width is `occlusion_width * dos_calculation_resolution`.
            occlusion_step (int): The step size for moving the occlusion window. Actual step size is `occlusion_step * dos_calculation_resolution`.
            dos_calculation_resolution (float): The resolution for DOS calculations.

        Raises:
            ValueError: When input values are not as expected.
            warnings.Warn: When the number of samplings is not greater than 500.
        """
        # Check the shape of dos_array
        numSamplings, numOrbitals, numChannels = dos_array.shape
        if numSamplings <= 500:
            warnings.warn("The number of samplings is not greater than 500.")
        if numOrbitals not in {1, 4, 9, 16}:
            raise ValueError("numOrbitals must be one of {1, 4, 9, 16}")
        if numChannels != 1:
            raise ValueError("numChannels must be 1")

        if dos_calculation_resolution <= 0:
            raise ValueError("DOS calculation resolution must be greater than zero.")

        # Check occlusion parameters
        if not isinstance(occlusion_step, int) or occlusion_step < 1:
            raise ValueError("Occlusion step must be an integer greater than 1.")
        if not isinstance(occlusion_width, int) or occlusion_width <= 0 or not ((occlusion_width - 1) / 2).is_integer():
            raise ValueError("(occlusion_width - 1)/2 must be a non-negative integer, and occlusion_width must be an integer.")

        self.dos_array = np.squeeze(dos_array, axis=2)  # squeeze shape to (numSamplings, numOrbitals)
        self.dos_calculation_resolution = dos_calculation_resolution
        self.occlusion_width = occlusion_width
        self.occlusion_step = occlusion_step

    def generate_occlusion_arrays(self) -> np.ndarray:
        """
        Generate a series of occlusion DOS arrays, where each array is created by occluding a region in the original DOS array (not going across orbitals).

        Returns:
            np.ndarray:
        """
        numSamplings, numOrbitals = self.dos_array.shape
        # Calculate total number of occlusions
        num_occlusions = ((numSamplings - self.occlusion_width) // self.occlusion_step) + 1
        if not float(num_occlusions).is_integer():
            raise ValueError(f"The resultant number of occlusions must be an integer. Current value is {num_occlusions}.")

        # Initialize a zero-filled array to store the occlusion arrays
        occlusion_arrays = np.zeros((num_occlusions, numSamplings, numOrbitals))



        return occlusion_arrays
