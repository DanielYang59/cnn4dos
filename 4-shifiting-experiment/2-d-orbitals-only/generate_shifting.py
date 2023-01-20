#!/usr/bin/env python3
# -*- coding: utf-8 -*-


shift_value = 0.1  # positive value: -X eV, X eV
elements = ["4-Ti", "4-V", "4-Cr", "4-Mn", "4-Fe", "4-Co", "4-Ni", "4-Cu"]
orbital_index = 6  # starts from zero: 0 s, 1 py, 2 pz, 3 px, 4 dxy, 5 dyz, 6 dz2, 7 dxz, 8 dx2-y2

dos_dir = "/Users/yang/Library/CloudStorage/OneDrive-QueenslandUniversityofTechnology/0-课题/3-DOS神经网络/1-模型和数据集/2-仅初态预测模型/final-model/dataset/feature_DOS/g-C3N4/3-CO_is"
model_checkpoint_dir = "/Users/yang/Library/CloudStorage/OneDrive-QueenslandUniversityofTechnology/0-课题/3-DOS神经网络/1-模型和数据集/2-仅初态预测模型/final-model/checkpoint"


import numpy as np
energy_array = np.linspace(-14, 6, 4000)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from sklearn.preprocessing import normalize


def shift_dos(shift_array, dos_array, energy_array, orbital_index):
    """Generate shifting experiment result.

    Args:
        shift_array (np.ndarray): _description_
        dos_array (np.ndarray): original dos array
        energy_array (np.ndarray): DOS energy array
        orbital_index (int): index of orbital to be shifted
        
    """
    # Check args
    assert isinstance(shift_array, np.ndarray)
    assert isinstance(dos_array, np.ndarray)
    assert isinstance(energy_array, np.ndarray)
    assert isinstance(orbital_index, int) and orbital_index in range(9)
    
    # Generate array of shifting arrays
    shifted_arrays = []
    for energy_shift in shift_array:
        # Zero pad to cover original energy range
        padded_orbital = pad_energy_and_dos_array(energy_array, np.copy(dos_array[:, orbital_index]), energy_shift)
        
        # Replace orbital DOS
        padded_dos = np.copy(dos_array)
        padded_dos[:, orbital_index] = padded_orbital
        
        shifted_arrays.append(padded_dos)

    return shifted_arrays


def pad_energy_and_dos_array(energy_array, dos_array, energy_shift):
    """_summary_

    Args:
        energy_array (_type_): _description_
        dos_array (_type_): _description_
        energy_shift (_type_): positive for shifting DOS to the right

    Returns:
        _type_: _description_
    """
    # Check args
    assert isinstance(energy_array, np.ndarray)
    assert isinstance(dos_array, np.ndarray)
    assert energy_array.shape == dos_array.shape
    assert isinstance(energy_shift, (int, float))
   
    
    # Copy original energy array
    shifted_energy_array = np.copy(energy_array)
    
    # Shift original energy array
    shifted_energy_array -= energy_shift
    
    # Crop DOS array according to shifted energy array
    ft = np.where(np.logical_and(energy_array >= shifted_energy_array[0], energy_array <= shifted_energy_array[-1]))  # generate filter
    
    dos_array = dos_array[ft]  # apply filter

    # Generate padding
    pad_width = energy_array.shape[0] - dos_array.shape[0]  # calculate padding size
    
    ## DOS array moved to left
    if energy_shift < 0:
        padded_dos_array = np.pad(dos_array, (0, pad_width), mode="constant", constant_values=0)  # pad on the right
        
    ## DOS array moved to right 
    elif energy_shift > 0:
        padded_dos_array = np.pad(dos_array, (pad_width, 0), mode="constant", constant_values=0)  # pad on the left
    else:
        padded_dos_array = dos_array


    assert energy_array.shape == padded_dos_array.shape
    return padded_dos_array


if __name__ == "__main__":
    # Print variables
    print(f"Starting shifting experiements...Shift range {shift_value} eV, orbital index {orbital_index}.")
    
    # Create storage directory
    os.makedirs(f"orbital_{orbital_index}", exist_ok=True)
    
    # Load model
    model = tf.keras.models.load_model(model_checkpoint_dir)
    
    # Generate shift array
    assert shift_value > 0
    shift_array = np.arange(-shift_value, shift_value + 0.005, 0.005)  # match DOS resolution

    # Loop through required elements
    for ele in elements:
        # Load original DOS and normalize
        dos_array = np.load(os.path.join(dos_dir, ele, "dos_up.npy"))
        dos_array = normalize(dos_array, axis=0, norm="max")
        
        # Generate reference energy (not shifted)
        reference_energy = model.predict(np.expand_dims(dos_array, axis=0), verbose=0).flatten()[0]
        
        # Generate shifted DOS arrays
        shifted_dos_arrays = shift_dos(shift_array, dos_array, energy_array, orbital_index)
        
        # Make predictions
        shifted_dos_arrays = np.stack(shifted_dos_arrays)
        predictions = np.array(model.predict(shifted_dos_arrays, verbose=0).flatten())
        
        # Reference to unshifted energy
        predictions -= reference_energy
        
        # Save shifting array
        np.save(os.path.join(f"orbital_{orbital_index}", f"{ele}.npy"), predictions)
    