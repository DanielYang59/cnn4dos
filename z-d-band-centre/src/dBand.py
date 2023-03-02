#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
from load_fermi_level import load_fermi_level


class dBand:
    def __init__(self, dosFile, fermi_level_dir, energy_range, fileType="numpy",):
        # Check args
        assert os.path.exists(dosFile)
        if fileType not in {"numpy", }:
            raise ValueError(f"DOS file type \"{fileType}\" is currently not supported.")
        assert isinstance(energy_range, (list, tuple)) and len(energy_range) == 2
        self.energy_range = energy_range       
         
        
        # Load DOS from file
        self.__load_dos(dosFile, fileType)
        
        self.nedos = self.dos_array.shape[0]
        self.numOrbitals = self.dos_array.shape[1]
        
        # Take d-band DOS
        self.__take_d_band()
        
        
        # Load fermi level
        self.fermi_level = load_fermi_level(dosFile, fermi_level_dir)
              
        
    def __calculate_band_moment(self, ordinal):
        """Calculate nth-order band moment.

        Args:
            ordinal (int): ordinal of d-band moment
            
        """
        # Check args
        assert isinstance(ordinal, int) and ordinal >= 1
        
        
        # Calculate
        
        
        
        
        
        
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
            
            
        assert d_band.shape[1] == 5
        self.d_band_array = d_band
    
    
    def calculate_d_band_centre(self, verbose=False):
        """Calculate d-band centre (reference to fermi level).

        Args:
            verbose (bool, optional): verbose. Defaults to False.

        Notes:
            1. The d-band center was calculated as the first moment of the projected d-band density of states on the surface atoms referenced to the Fermi level, and the mean squared d-band width was calculated as the second moment.(J. Chem. Phys. 120 (2004) 10240)".
            2. ref: https://sites.psu.edu/anguyennrtcapstone/example-calculation/how-to-calculate-the-d-band-center/
            3. ref: http://theory.cm.utexas.edu/forum/viewtopic.php?t=649

        Returns:
            np.float64: _description_
            
        """
        # Calculate d-band centre (referenced to fermi level)
        d_band_centre = self.__calculate_band_moment(ordinal=1)
        
        if verbose:
            print(f"d-band centre is {d_band_centre}.")
        
        
        return d_band_centre
    

# Test area
if __name__ == "__main__":
    
    example_dos_file = "../../0-dataset/feature_DOS/g-C3N4/3-CO_is/4-Co/dos_up_57.npy"
    d_band = dBand(dosFile=example_dos_file, fileType="numpy",
                   fermi_level_dir="../../0-dataset/z-supporting-info/fermi_level",
                   energy_range=[-14, 6],
                   )
    
    d_band.calculate_d_band_centre(verbose=True)
     