#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import warnings


class dBand:
    def __init__(self, dosFile, fileType="numpy"):
        # Check args
        assert os.path.exists(dosFile)
        if fileType not in {"numpy", }:
            raise ValueError(f"DOS file type \"{fileType}\" is currently not supported.")
        
        # Load DOS from file
        self.__load_dos(dosFile, fileType)
        
        self.nedos = self.dos_array.shape[0]
        self.numOrbitals = self.dos_array.shape[1]
        
        
        
    def __load_dos(self, dosFile, fileType):
        """Load DOS file to numpy array, expecting DOS in shape (NEDOS, numOrbitals).

        Args:
            dosFile (str): path to DOS file
            fileType (str): DOS file type
            
        Attrib:
            dos_array (np.ndarray): DOS array in shape (NEDOS, numOrbitals)

        """
        # Load DOS file
        if fileType == "numpy":
            dos_array = np.load(dosFile)
            
        else:
            # Covert to numpy array (if fileType isn't numpy array)
            pass
            
        
        # Check DOS array shape
        numOrbitals = dos_array.shape[1]
        if numOrbitals not in {1, 4, 9, 16}:
            raise ValueError(f"Illegal number of DOS orbitals \"{numOrbitals}\" found! Expecting DOS in shape (NEDOS, numOrbitals).")
        if dos_array.ndim != 2:
            raise ValueError("Expecting DOS in shape (NEDOS, numOrbitals).")
        

        assert isinstance(dos_array, np.ndarray)
        self.dos_array = dos_array
    
    
    def __take_d_band(self):
        """Take d-band from DOS array.
        
        Attrib:
            d_band_array (np.ndarray): d-band DOS array in shape (NEDOS, 5)
            
        """
        # Check attrib
        assert hasattr(self, "dos_array")
        
        
        # Take d orbitals from DOS array
        if self.numOrbitals in {1, 4}:
            d_band = np.zeros((self.nedos, 5))  # generate zero array when no d-orbital presents
  
  
        elif self.numOrbitals in {9, 16}:
            d_band = np.copy(self.dos_array)[:, 4:9]
        
        else:
            raise ValueError(f"Illegal number of orbitals \"{self.numOrbitals}\".")
            
        
        assert d_band.shape[2] == 5
        self.d_band_array = d_band
        
    
    def calculate_band_centre(self):
        pass
    
    
    def calculate_band_width(self):
        pass


# Test area
if __name__ == "__main__":
    example_dos_file = "../../0-dataset/feature_DOS/g-C3N4/3-CO_is/4-Co/dos_up_57.npy"
    
    d_band = dBand(dosFile=example_dos_file)
    