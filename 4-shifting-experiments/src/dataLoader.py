#!/bin/usr/python3
# -*- coding: utf-8 -*-

import numpy as np
import yaml
from pathlib import Path
import warnings

class DataLoader:

    def __init__(self):
        pass

    def load_config(self, config_path: str) -> dict:
        """
        Load the configuration YAML file.

        Args:
            config_path (str): The path to the config.yaml file.

        Returns:
            dict: The configuration data as a Python dictionary.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"The specified file {config_path} does not exist.")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def load_and_preprocess_adsorbate_dos(self, filepath: str) -> np.ndarray:
        """
        Load and preprocess the adsorbate DOS array.

        Args:
            filepath (str): The path to the .npy file.

        Returns:
            np.ndarray: The preprocessed adsorbate DOS array of shape (numSamplings, numOrbitals, numChannels).

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If numOrbitals is not in {1, 4, 9, 16}.

        Note:
            The original shape of the adsorbate DOS array in the file should be (numChannels, numSamplings, numOrbitals).
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"The specified file {filepath} does not exist.")

        adsorbate_dos = np.load(filepath)
        adsorbate_dos = np.transpose(adsorbate_dos, (1, 2, 0))

        numSamplings, numOrbitals, numChannels = adsorbate_dos.shape

        if numOrbitals not in {1, 4, 9, 16}:
            raise ValueError("numOrbitals must be one of {1, 4, 9, 16}")

        if numSamplings <= 500:
            warnings.warn("The number of samplings is not greater than 500.")

        if numChannels >= 20:
            warnings.warn("The number of channels is greater than 20.")

        return adsorbate_dos

    def load_unshifted_dos(self, filepath: str) -> np.ndarray:
        """
        Load the unshifted DOS array.

        Args:
            filepath (str): The path to the .npy file containing the unshifted DOS array.

        Returns:
            np.ndarray: The preprocessed DOS array of shape (numSamplings, numOrbitals, numChannels=1).

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If numOrbitals is not in {1, 4, 9, 16}.

        Note:
            The original shape of the DOS array in the file should be (numSamplings, numOrbitals).
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"The specified file {filepath} does not exist.")

        unshifted_dos = np.load(filepath)
        numSamplings, numOrbitals = unshifted_dos.shape
        unshifted_dos = np.expand_dims(unshifted_dos, axis=-1)  # Reshape to (numSamplings, numOrbitals, 1)

        if numOrbitals not in {1, 4, 9, 16}:
            raise ValueError("numOrbitals must be one of {1, 4, 9, 16}")

        if numSamplings <= 500:
            warnings.warn("The number of samplings is not greater than 500.")

        return unshifted_dos
