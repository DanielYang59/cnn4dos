import csv
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
    
    
    def load_feature(self, path, substrates, adsorbates, dos_filename, keysep=":", states={"is", "fs"}):
        """Load DOS dataset feature from given list of dirs.

        Args:
            path (str): path to dataset dir
            substrates (list): list of substrates to load
            adsorbates (list): list of adsorbates to load
            filename (str): name of the DOS file under each dir
            keysep (str): separator for dir and project name in dataset dict

        Notes:
            1. DOS array in (NEDOS, orbital) shape
            2. feature dict key is "{substrate}{keysep}{adsorbate}{keysep}is/fs" (is for initial state, fs for final state)

        """
        # Check args
        assert os.path.isdir(path)
        assert isinstance(substrates, list)
        assert isinstance(adsorbates, list)
        assert isinstance(dos_filename, str) and dos_filename.endswith(".npy")
        self.substrates = substrates
        self.adsorbates = adsorbates
        
        
        # Import DOS as numpy array
        feature_data = {}
        for sub in substrates:
            for ads in adsorbates:
                for state in states:
                    
                    # Compile path 
                    directory = os.path.join(path, sub, f"{ads}_{state}")
                    assert os.path.isdir(directory)
                    
                    # Loop through all directories to load DOS
                    for folder in os.listdir(directory):
                        if os.path.isdir(os.path.join(directory, folder)) and os.path.exists(os.path.join(directory, folder, dos_filename)):
                            # Compile dict key as "{substrate}{keysep}{adsorbate}{keysep}{state}"
                            key = f"{sub}{keysep}{ads}{keysep}{state}{keysep}{folder}"

                            # Parse data as numpy array into dict
                            arr = np.load(os.path.join(directory, folder, dos_filename))
                            feature_data[key] = arr  # shape (4000, 9)
                    
        
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


    def append_adsorbate(self, adsorbate_dos_dir, dos_name="dos_up_adsorbate.npy"):
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
