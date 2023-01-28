#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


class Dataset:
    """Dataset class for loading and manipulating DOS dataset for Deep Learning.
    
    Attributes:
        feature (dict): DOS feature, key is "{substrate}{keysep}{adsorbate}{keysep}is/fs", value is DOS array
        numFeature (int): total number of samples
        featureKeySep (str): separator used in dict keys
        substrates (list):
        adsorbates (list):
    
    """
    def __init__(self) -> None:
        pass
    
    
    def load_feature(self, path, substrates, adsorbates, centre_atoms, states=("is", "fs"), spin="up", load_augment=False, augmentations=None, keysep=":", remove_ghost=False):
        """Load DOS dataset feature from given list of dirs.

        Args:
            path (str): path to dataset dir
            substrates (list): list of substrates to load
            adsorbates (list): list of adsorbates to load
            centre_atoms (dict): centre atom index dict (index starts from 1)
            filename (str): name of the DOS file under each dir
            keysep (str): separator for dir and project name in dataset dict
            states (tuple): list of states, "is" for initial state, "fs" for final state
            spin (str): load spin "up" or "down" DOS, or "both"
            load_augment (bool): load augmentation data or not, augmented substrate should end with "_aug"
            augmentations (list): list of augmentation distances
            remove_ghost (bool): remove ghost state (first point of NEDOS)
            
        Notes:
            1. DOS array in (NEDOS, orbital) shape
            2. feature dict key is "{substrate}{keysep}{adsorbate}{keysep}is/fs" (is for initial state, fs for final state)
            3. Spin up DOS should be named "dos_up.npy", down "dos_down.npy"

        """
        # Check args
        assert os.path.isdir(path)
        assert isinstance(substrates, list)
        assert isinstance(adsorbates, list)
        assert isinstance(centre_atoms, dict)
        for index in centre_atoms.values():
            assert isinstance(index, int) and index >= 1
        for state in states:
            assert state in {"is", "fs"}
        assert spin in {"up", "down", "both"}
        assert isinstance(load_augment, bool)
        assert isinstance(remove_ghost, bool)
            
        # Append augmentation to substrates if required
        if load_augment:
            assert isinstance(augmentations, list)
            for i in augmentations:
                assert isinstance(i, str)
            substrates.extend([f"{i}_aug" for i in substrates])
            print(f"Augmentation data would be loaded: {augmentations}")
        
        # Update attrib
        self.substrates = substrates
        self.adsorbates = adsorbates
        
        
        # Import DOS as numpy array
        feature_data = {}
        for sub in substrates:
            # Get centre atom index from dict
            centre_atom_index = centre_atoms[sub]
            
            for ads in adsorbates:
                for state in states:
                    
                    # Compile path
                    directory = os.path.join(path, sub, f"{ads}_{state}")
                    assert os.path.isdir(directory)
                    
                    # Loop through all directories to load DOS
                    for folder in os.listdir(directory):
                        if os.path.isdir(os.path.join(directory, folder)) and (os.path.exists(os.path.join(directory, folder, f"dos_up_{centre_atom_index}.npy")) or os.path.exists(os.path.join(directory, folder, f"dos_down_{centre_atom_index}.npy"))):
                            # Do augmentation distance check for augmented data
                            if sub.endswith("_aug") and folder.split("_")[-1] not in augmentations:
                                continue
                            

                            # Compile dict key as "{substrate}{keysep}{adsorbate}{keysep}{state}"
                            key = f"{sub}{keysep}{ads}{keysep}{state}{keysep}{folder}"

                            # Parse data as numpy array into dict
                            ## Load spin up
                            if spin == "up":
                                arr = np.load(os.path.join(directory, folder, f"dos_up_{centre_atom_index}.npy"))
                            elif spin == "down":
                                arr = np.load(os.path.join(directory, folder, f"dos_down_{centre_atom_index}.npy"))
                            else:  # load both spin-up and down
                                arr_up = np.load(os.path.join(directory, folder, f"dos_up_{centre_atom_index}.npy"))  # (NEDOS, numOrbital)
                                arr_down = np.load(os.path.join(directory, folder, f"dos_down_{centre_atom_index}.npy")) # (NEDOS, numOrbital)
                                arr = np.stack([arr_up, arr_down], axis=2)  # (NEDOS, numOrbital, 2)
                            
                            
                            # Zero out first point along NEDOS axis to remove "ghost state"
                            if remove_ghost:
                                arr[0] = 0
                            
                            # Update dict value
                            feature_data[key] = arr  # shape (NEDOS, numOrbital)
                    
        
        # Update attrib
        self.feature = feature_data
        self.numFeature = len(feature_data)
        self.featureKeySep = keysep

 
    def scale_feature(self, mode):
        """Scale feature arrays. 
        
        Args:
            mode (str): scaling mode
            
        Notes:
            arr shape: (NEDOS, orbital)
            
        """
        # Check args
        assert mode in {"normalization", "none"}

        # Loop through dataset and perform scaling
        for key, arr in self.feature.items():
            if mode == "normalization":
                # Perform normalize
                self.feature[key] = normalize(arr, axis=0, norm="max")


    def load_label(self, label_dir):
        """Load labels based on names of feature files.

        Args:
            label_dir (str): label csv files directory.

        """
        # Check args
        assert os.path.isdir(label_dir)

        # Load label csv as pd DataFrame
        labels_source = {}
        for file in os.listdir(label_dir):
            if file.endswith(".csv") and not file.startswith("."):
                labels_source[file.replace(".csv", "")] = pd.read_csv(os.path.join(label_dir, file))

        
        # Fetch label from source dict
        labels = {}
        for key in self.feature:  # key is "{substrate}{keysep}{adsorbate}{keysep}{state}"
            # Unpack key
            substrate, adsorbate, state, project = key.split(self.featureKeySep)

            # Find label from dataframe
            df = labels_source[f"{substrate}_{state}"]
            df.index = df.iloc[:, 0] # set first column as headers
            try:
                labels[key] = float(df.loc[project][adsorbate])
            except KeyError:
                raise KeyError(f"Label for key:\"{key}\" not found.")
            
        self.label = labels


    def append_adsorbate_DOS(self, adsorbate_dos_dir, dos_name="dos_up_adsorbate.npy"):
        """Append adsorbate DOS to metal DOS.

        Args:
            adsorbate_dos_dir (str): adsorbate DOS directory.
            dos_name (str, optional): name of adsorbate DOS. Defaults to "dos_up_adsorbate.npy".
            
        """
        # Check args
        assert os.path.isdir(adsorbate_dos_dir)
        
        # Loop through dataset and append adsorbate DOS
        for key, arr in self.feature.items():
            # Get adsorbate name
            mol_name = key.split(":")[1]
            # Load adsorbate DOS
            mol_dos_arr = np.load(os.path.join(adsorbate_dos_dir, mol_name, dos_name))
            
            # Append to original DOS
            arr = np.expand_dims(arr, axis=0)  # reshape original DOS from (4000, 9) to (1, 4000, 9)
            arr = np.concatenate([arr, mol_dos_arr])
            
            # Swap (6, 4000, 9) to (4000, 9, 6)
            arr = np.swapaxes(arr, 0, 1)
            arr = np.swapaxes(arr, 1, 2)
            
            # Update feature dict
            self.feature[key] = arr
