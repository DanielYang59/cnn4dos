#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""."""

__author__ = "Haoyu Yang"
__credits__ = ["Qiankun Wang", "Yanwei Guan"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Haoyu Yang"
__email__ = "h51.yang@hdr.qut.edu.au"

"""
TODO:
2. occlusion when adsorbate DOS appended (don't loop adsorbate DOS)

"""



# Configs
model_path = "../1-model-and-training/2-best-model/model"
normalize_dos = False
append_molecule = True
adsorbate_dos_dir = "../0-dataset/feature_DOS/adsorbate-DOS"


masker_width = 1
masker = 0
dos_array_file = "dos_up.npy"
output_arr_name = "occlusion.npy"


import os, sys
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from sklearn.preprocessing import normalize


def generate_occlusion_arrays(source_dos_array, masker_width, masker):
    """Generate array of occluded arrays from source DOS array.

    Args:
        source_dos_array (np.ndarray): source DOS array where occlusion would be applied
        masker_width (int): masker width
        masker ((int, float), optional): value used for masking

    Returns:
        np.ndarray: array of masked arrays
        
    """
    # Check args
    assert isinstance(source_dos_array, np.ndarray)
    assert isinstance(masker_width, int) and masker_width >= 1
    assert isinstance(masker, (int, float))
    
    # Calculate padding size
    padding_size = (masker_width - 1) / 2
    assert float(padding_size).is_integer()  # padding size should be integer
    padding_size = int(padding_size)

    # Pad source DOS array
    if padding_size == 0:  # width is 1, no need for padding
        padded_dos_array = source_dos_array
        
    else:
        padded_dos_array = []
        for orbital_index in range(source_dos_array.shape[1]):
            orbital_arr = source_dos_array[:, orbital_index]
            orbital_arr = np.pad(orbital_arr, (padding_size, padding_size), "constant", constant_values=(0, 0))
            padded_dos_array.append(orbital_arr)
        padded_dos_array = np.stack(padded_dos_array).transpose()
    
        assert padded_dos_array.shape[0] == source_dos_array.shape[0] + masker_width - 1
    
    
    # Create empty array to host results
    result_arrays = np.zeros(source_dos_array.shape, dtype=object)
    
    # Perform masking for each element in each orbital
    for orbital_index in range(source_dos_array.shape[1]):
        for x_index in range(source_dos_array.shape[0]):
             masked_array = mask_dos_array(padded_arr=padded_dos_array, masking_position=(x_index, orbital_index), masker_width=masker_width, padding_size=padding_size, masker=masker)
             assert masked_array.shape == source_dos_array.shape
             result_arrays[x_index, orbital_index] = masked_array
    
    return result_arrays


def mask_dos_array(padded_arr, masking_position, masker_width, padding_size, masker):
    """Mask padded DOS array at given index (row_index, col_index).

    Args:
        padded_arr (np.ndarray): padded DOS array to be masked
        masking_position (tuple, list): index to apply masker
        masker_width (int): masker width
        padding_size (int): padding size
        masker (int, optional): value used for masking. Defaults to 0.

    Returns:
        np.ndarray: DOS array after part being masked
        
    """
    # Check args
    assert isinstance(padded_arr, np.ndarray)
    assert isinstance(masking_position, (tuple, list)) and len(masking_position) == 2
    assert isinstance(masker_width, int) and masker_width >= 1
    assert isinstance(padding_size, int) and padding_size >= 0
    
    # Unpack row/col index
    row_index = masking_position[0]
    col_index = masking_position[1]
    
    # Perform masking at nominated position
    result_array = np.copy(padded_arr)
    
    if masker_width == 1:
        if masker == 0:  # full masking
            result_array[row_index, col_index] = masker
        else:  # constant masking
            result_array[row_index, col_index] -=  masker 
    
    else:
        # Apply masking at specific position
        if masker == 0:  # full masking
            result_array[row_index: row_index + masker_width, col_index] = masker 
        else:  # constant masking
            result_array[row_index: row_index + masker_width, col_index] -= masker 

        # Crop padded array to original shape
        result_array = result_array[padding_size: -padding_size, :]
    
    return result_array


if __name__ == "__main__":
    # Check args
    assert os.path.exists(dos_array_file) and dos_array_file.endswith(".npy")
    assert isinstance(masker_width, int) and masker_width >= 1
    assert masker == 0  # occlusion only
    print(f"Performing occlusion experiments with masker_width {masker_width}, DOS normalized.")
    
    # Compile path for model and adsorbate DOS
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
    adsorbate_dos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), adsorbate_dos_dir) 
    assert os.path.isdir(model_path)
    assert os.path.isdir(adsorbate_dos_dir)
    
    
    # Import source DOS array in shape (NEDOS, orbital)
    source_dos_array = np.load(dos_array_file)
    if source_dos_array.shape[1] > 16:
        raise ValueError(f"Source array has shape {source_dos_array.shape}")
    ## Append molecule DOS
    if append_molecule:
        print("WARNING! Molecule DOS would be appended.")
        # Get adsorbate name
        mol_name = os.getcwd().split(os.sep)[-2].split("_")[0]
        
        # Load adsorbate DOS
        mol_dos_arr = np.load(os.path.join(adsorbate_dos_dir, mol_name, "dos_up_adsorbate.npy"))
        
        # Append to original DOS
        source_dos_array = np.expand_dims(source_dos_array, axis=0)  # reshape original DOS from (4000, 9) to (1, 4000, 9)
        source_dos_array = np.stack([source_dos_array, mol_dos_arr], axis=2)
        
        
        # Swap (6, 4000, 9) to (4000, 9, 6)
        source_dos_array = np.swapaxes(source_dos_array, 0, 1)
        source_dos_array = np.swapaxes(source_dos_array, 1, 2)
        
         
    ## Normalize source DOS array 
    if normalize_dos:
        print("WARNING! DOS would be normalized.")
        scaled_arr = []

        for channel_index in range(source_dos_array.shape[2]):
            channel = normalize(source_dos_array[:, :, channel_index], axis=0, norm="max")
            scaled_arr.append(channel)
                                
        # Update source array
        scaled_arr = np.stack(scaled_arr, axis=2)


    # Import model
    model = tf.keras.models.load_model(os.path.join(os.path.realpath(os.path.dirname(__file__)), model_path))
    ## Predict original label
    original_label = model.predict(np.expand_dims(source_dos_array, axis=0), verbose=0)[0][0]
    
    
    # Generate occlusion arrays
    occlusion_arrays = generate_occlusion_arrays(source_dos_array, masker_width, masker)
    
    
    # Generate predicted labels
    predicted_labels = []
    ## Loop through each orbital and make prediction
    for orbital_index in range(occlusion_arrays.shape[1]):
        col_of_arrays = np.stack(occlusion_arrays[:, orbital_index])
        # Use model for prediction
        print(f"Making prediction in orbital {orbital_index}")
        predictions = model.predict(col_of_arrays, verbose=0).flatten()
        
        # Subtract original label
        predicted_labels.append(predictions - original_label)
    
    predicted_labels = np.stack(predicted_labels).transpose()  # tranpose (orbital, NEDOS) to (NEDOS, orbital)
    
    # Save predicted labels to local file
    np.save(output_arr_name, predicted_labels)
