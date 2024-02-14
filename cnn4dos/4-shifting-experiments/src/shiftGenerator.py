"""Generate eDOS shifted arrays."""

import warnings
from typing import List

import numpy as np


class ShiftGenerator:
    """
    Class for generating Density of States (DOS) arrays with shifting.
    """

    def __init__(
        self,
        dos_array: np.ndarray,
        shifting_range: list,
        shifting_step: float,
        shifting_orbitals: list,
        dos_calculation_resolution: float,
    ) -> None:
        """
        Initialize the ShiftGenerator.

        Args:
            dos_array (np.ndarray): The pre-processed DOS array of shape (numSamplings, numOrbitals, 1).
            shifting_range (list): A list containing the start and end of the shifting range.
            shifting_step (float): The step length for each shift.
            shifting_orbitals (list): A list containing the indices of the orbitals to be shifted.
            dos_calculation_resolution (float): The energy resolution along the numSamplings axis of the DOS array.

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

        if dos_calculation_resolution <= 0:
            raise ValueError("DOS calculation resolution must be greater than zero.")

        # Check shifting parameters
        start, end = shifting_range
        if start >= end:
            raise ValueError(
                "The start of the shifting range should be smaller than the end."
            )
        if shifting_step <= 0:
            raise ValueError("shift_step should be greater than zero.")
        if not self._is_multiple_of(number=(end - start), base=shifting_step):
            raise ValueError(
                "The shifting range (end - start) should be a multiple of shift_step."
            )
        if len(set(shifting_orbitals)) != len(shifting_orbitals):
            raise ValueError("Duplicate orbital indices found in shifting_orbitals.")
        if any((idx < 0 or idx >= numOrbitals) for idx in shifting_orbitals):
            raise ValueError(
                f"Invalid orbital indices in shifting_orbitals. Valid range is [0, {numOrbitals-1}]"
            )
        if not self._is_multiple_of(shifting_step, dos_calculation_resolution):
            raise ValueError(
                "shifting_step must be a multiple of dos_calculation_resolution."
            )

        self.dos_array = dos_array
        self.shifting_range = shifting_range
        self.shift_step = shifting_step
        self.shifting_orbitals = shifting_orbitals
        self.dos_calculation_resolution = dos_calculation_resolution

    def _is_multiple_of(self, number, base) -> bool:
        """
        Check if a given float number is a multiple of another float base,
        considering numerical inaccuracies.

        Args:
            number (float): The number to check.
            base (float): The base to divide by.

        Returns:
            bool: True if 'number' is a multiple of 'base', False otherwise.
        """
        quotient = number / base
        return np.isclose(quotient, round(quotient))

    def generate_shifted_arrays(self) -> List[np.ndarray]:
        """
        Generate a series of shifted DOS arrays.

        Returns:
            list: A list of shifted DOS arrays.
        """
        start, end = self.shifting_range
        shift_values = np.arange(start, end + self.shift_step, self.shift_step)
        shifted_arrays = []

        for shift_value in shift_values:
            # Create a copy to store the shifted array
            shifted_dos = np.copy(self.dos_array)

            # Calculate the number of indices to shift based on dos_calculation_resolution
            num_indices_to_shift = int(
                abs(shift_value) / self.dos_calculation_resolution
            )

            if num_indices_to_shift >= shifted_dos.shape[0]:
                raise ValueError(
                    f"Skipping shift value {shift_value} as it is greater or equal to the data shape."
                )

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
                    padded_data = np.concatenate([padding, shifted_data])
                else:  # shift_value < 0
                    padded_data = np.concatenate([shifted_data, padding])

                # Assign the padded data back to the array
                shifted_dos[:, orbital_idx, 0] = padded_data.flatten()
            shifted_arrays.append(shifted_dos)

        return shifted_arrays
