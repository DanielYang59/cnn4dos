#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""."""

model_checkpoint = "/Users/yang/Library/CloudStorage/OneDrive-QueenslandUniversityofTechnology/0-课题/3-DOS神经网络/1-模型和数据集/2-仅初态预测模型/final-model/checkpoint"

masker_width = 1
masker = 0.001

dos_array_file = "dos_up.npy"
output_arr_name = "perturbation.npy"


import os
import copy
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from sklearn.preprocessing import normalize


def generate_perturbation_arrays(source_dos_array, masker_width, masker, norms):
    """Generate array of occluded arrays from source DOS array.

    Args:
        source_dos_array (np.ndarray): source DOS array where occlusion would be applied
        masker_width (int): masker width
        masker ((int, float), optional): value used for masking
        norms (np.ndarray): norms of each orbital

    Returns:
        np.ndarray: array of masked arrays
        
    """
    # Check args
    assert isinstance(source_dos_array, np.ndarray)
    assert isinstance(masker_width, int) and masker_width >= 1
    assert isinstance(masker, (int, float))
    assert norms.shape[0] == source_dos_array.shape[1]
    
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
            # Take norms into account
            normed_masker = copy.copy(masker)
            normed_masker = normed_masker / norms[orbital_index]
            
            masked_array = mask_dos_array(padded_arr=padded_dos_array, masking_position=(x_index, orbital_index), masker_width=masker_width, padding_size=padding_size, masker=normed_masker)
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
    assert os.path.isdir(model_checkpoint)
    assert isinstance(masker_width, int) and masker_width >= 1
    print(f"Performing perturbation experiment with masker_width {masker_width} and masker {masker}, DOS normalized.")
    
    # Import source DOS array in shape (NEDOS, orbital)
    source_dos_array = np.load(dos_array_file)
    if source_dos_array.shape[1] > 16:
        raise ValueError(f"Source array has shape {source_dos_array.shape}")
    ## Normalize source DOS array 
    source_dos_array, norms = normalize(source_dos_array, axis=0, norm="max", return_norm=True)
    
    # Import DL model
    model = tf.keras.models.load_model(model_checkpoint)
    ## Predict original label
    original_label = model.predict(np.expand_dims(source_dos_array, axis=0), verbose=0)[0][0]

    
    # Generate pertubation arrays
    perturbation_arrays = generate_perturbation_arrays(source_dos_array, masker_width, masker, norms) #DEBUG: norm take into account
    
    
    # Generate predicted labels
    predicted_labels = []
    ## Loop through each orbital and make prediction
    for orbital_index in range(perturbation_arrays.shape[1]):
        col_of_arrays = np.stack(perturbation_arrays[:, orbital_index])
        # Use model for prediction
        print(f"Making prediction in orbital {orbital_index}")
        predictions = model.predict(col_of_arrays, verbose=0).flatten()
        
        # Subtract original label
        predicted_labels.append(predictions - original_label)
    
    predicted_labels = np.stack(predicted_labels).transpose()  # tranpose (orbital, NEDOS) to (NEDOS, orbital)
    
    
    # Divide by effective masker value (number of electrons) 
    predicted_labels = predicted_labels / masker
     
    # Save predicted labels to local file
    np.save(output_arr_name, predicted_labels)
