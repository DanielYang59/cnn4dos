#!/usr/bin/env python3
# -*- coding: utf-8 -*-


working_dir = "."
shift_value = 0.1  # positive value: -X eV, X eV
model_path = "../../1-model-and-training/2-best-model/model"
adsorbate_dos_dir = "../../0-dataset/feature_DOS/adsorbate-DOS"
normalize_dos = False
append_molecule = True
remove_ghost_state = True


import numpy as np
energy_array = np.linspace(-14, 6, 4000)


import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from sklearn.preprocessing import normalize


def shift_dos(shift_array, dos_array, energy_array):
    """Generate shifting experiment result.

    Args:
        shift_array (np.ndarray): _description_
        dos_array (np.ndarray): original dos array
        energy_array (np.ndarray): DOS energy array
        
    """
    # Check args
    assert isinstance(shift_array, np.ndarray)
    assert isinstance(dos_array, np.ndarray)
    assert isinstance(energy_array, np.ndarray)
    
    # Generate array of shifting arrays
    shifted_arrays = np.zeros((9, shift_array.shape[0]), dtype=object)
    
    ## Loop through all orbitals
    for orbital_index in range(9):        
        for energy_shift_index, energy_shift in enumerate(shift_array):
            # Zero pad to cover original energy range
            padded_orbital = pad_energy_and_dos_array(energy_array, np.copy(dos_array[:, orbital_index]), energy_shift)
            
            # Replace orbital DOS
            padded_dos = np.copy(dos_array)
            padded_dos[:, orbital_index] = padded_orbital

            # Put into corresponding location
            shifted_arrays[orbital_index, energy_shift_index] = padded_dos
        
    return shifted_arrays


def pad_energy_and_dos_array(energy_array, dos_array, energy_shift):
    """_summary_

    Args:
        energy_array (np.ndarray): _description_
        dos_array (np.ndarray): _description_
        energy_shift (float, int): positive for shifting DOS to the right

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


def combine_metal_and_adsorbate_DOS(metal_dos, adsorbate_dos):
    """Combine metal and adsorbate DOS.

    Args:
        metal_dos (np.ndarray): metal DOS in shape (NEDOS, numOrbitals)
        adsorbate_dos (np.ndarray): adsorbate DOS in shape (numChannels, NEDOS, numOrbitals) 

    Returns:
        np.ndarray: combined DOS arrays in shape (NEDOS, numOrbitals, numChannels)
        
    """
    # Check two DOS arrays shape
    assert len(metal_dos.shape) == 2  # shape in (NEDOS, numOrbitals)
    assert len(adsorbate_dos.shape) == 3  # shape in (numChannels, NEDOS, numOrbitals)
    
    # Append to original DOS
    metal_dos = np.expand_dims(metal_dos, axis=0)  # reshape original DOS from (4000, 9) to (1, 4000, 9)
    combined_dos_array = np.concatenate([metal_dos, adsorbate_dos], axis=0)
    
    
    # Swap (6, 4000, 9) to (4000, 9, 6)
    combined_dos_array = np.swapaxes(combined_dos_array, 0, 1)
    combined_dos_array = np.swapaxes(combined_dos_array, 1, 2)

    return combined_dos_array


if __name__ == "__main__":
    # Print variable
    print(f"Starting shifting experiments...Shift value {shift_value} eV.")
    
    # Compile path for model and adsorbate DOS 
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
    adsorbate_dos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), adsorbate_dos_dir) 
    assert os.path.isdir(model_path)
    assert os.path.isdir(adsorbate_dos_dir)
    
    # Load model
    model = tf.keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path))
    
    # Generate shift array
    assert shift_value > 0
    shift_array = np.arange(-shift_value, shift_value + 0.005, 0.005)  # match DOS resolution


    # Load original DOS and normalize
    dos_array = np.load(os.path.join(working_dir, "dos_up.npy"))
    ## Remove ghost state if required
    if remove_ghost_state:
        print("WARNING! Ghost state will be removed.")
        dos_array[0] = 0.0
        
    ## Load molecule DOS if required
    if append_molecule:
        print("WARNING! Molecule DOS would be appended.")
        # Load adsorbate DOS
        mol_name = os.getcwd().split(os.sep)[-2].split("_")[0]
        mol_dos_arr = np.load(os.path.join(adsorbate_dos_dir, mol_name, "dos_up_adsorbate.npy"))
    
    
    # if normalize_dos:
    #     dos_array = normalize(dos_array, axis=0, norm="max")
    
    # Generate reference energy (not shifted)
    reference_energy = model.predict(np.expand_dims(combine_metal_and_adsorbate_DOS(dos_array, mol_dos_arr), axis=0), verbose=0)[0][0]
    
    
    # Generate shifted DOS arrays
    shifted_dos_arrays = shift_dos(shift_array, dos_array, energy_array)
    
    # Make predictions with model
    prediction_arr = []
    ## Loop through each orbital
    for orbital_index in range(9):
        # Take orbital data
        orbital_array = shifted_dos_arrays[orbital_index, :]

        # Unpack nested arrays
        orbital_array = np.stack(orbital_array)
        
         # Append molecule DOS if requested
        if append_molecule:
            # Reshape (4000, 4000, 9) to (4000, 4000, 9, 1)
            orbital_array = np.expand_dims(orbital_array, axis=-1)
            
            # Reshape molecule DOS to shape (1, 4000, 9, 5)
            transposed_mol_dos_arr = np.swapaxes(np.copy(mol_dos_arr), 0, 1)
            transposed_mol_dos_arr = np.swapaxes(transposed_mol_dos_arr, 1, 2)
            
            transposed_mol_dos_arr = np.expand_dims(transposed_mol_dos_arr, axis=0)
            
            # Append molecule array to each source array
            orbital_array = np.concatenate([orbital_array, np.tile(transposed_mol_dos_arr, (orbital_array.shape[0], 1, 1, 1))], axis=-1)
        
        
        orbital_predictions = np.array(model.predict(orbital_array, verbose=0).flatten())
        
        # Collect data
        prediction_arr.append(orbital_predictions)
      
    # Pack data
    prediction_arr = np.stack(prediction_arr)
    
    # Reference to unshifted energy
    prediction_arr -= reference_energy
    
    # Save shifting results
    np.save(f"shifting_{shift_value}.npy", prediction_arr)
