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
              
    
    def __calculate_band_moment(self, single_dos_orbital, energy_array, ordinal):
        """Calculate nth-order band moment.

        Args:
            single_dos_orbital (np.ndarray): merged five orbitals of d-band
            energy_array (np.ndarray): energy array referred to fermi level
            ordinal (int): ordinal of d-band moment
            
        """
        # Check args
        assert isinstance(ordinal, int) and ordinal >= 1
        assert single_dos_orbital.ndim == 1
        
        
        # Calculate band moment
        density = np.copy(single_dos_orbital) ** ordinal
        
        numerator = np.trapz(y=(density * np.copy(energy_array)),
                             dx=(self.energy_range[1] - self.energy_range[0]) / self.nedos
                             )
        
        denominator = np.trapz(y=density,
                               dx=(self.energy_range[1] - self.energy_range[0]) / self.nedos
                               ) 
        
        return numerator / denominator
        
    
    def __calculate_band_centre(self, single_dos_orbital, energy_array):
        """Calculate band centre of a single DOS orbital.

        Args:
            single_dos_orbital (np.ndarray): merged five orbitals of d-band
            energy_array (np.ndarray): energy array referred to fermi level
            
        Notes:
            1. This method is deprecated. Use the more general "__calculate_band_moment" method instead whenever possible.
            2. Ref: https://sites.psu.edu/anguyennrtcapstone/example-calculation/how-to-calculate-the-d-band-center/
            
        """
        # Check DOS shape
        assert single_dos_orbital.ndim == 1

        
        # Calculate band centre
        numerator = np.trapz(y=(np.copy(single_dos_orbital) * np.copy(energy_array)),
                             dx=(self.energy_range[1] - self.energy_range[0]) / self.nedos
                             )
        
        denominator = np.trapz(y=single_dos_orbital,
                               dx=(self.energy_range[1] - self.energy_range[0]) / self.nedos
                               ) 
        
        return numerator / denominator 
    
        
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
    
    
    def calculate_d_band_centre(self, merge_suborbitals=True, verbose=False):
        """Calculate d-band centre (reference to fermi level).

        Args:
            merge_suborbitals (bool, optional): sum five d-suborbitals. Defaults to False.
            verbose (bool, optional): verbose. Defaults to False.

        Notes:
            1. The d-band center was calculated as the first moment of the projected d-band density of states on the surface atoms referenced to the Fermi level, and the mean squared d-band width was calculated as the second moment.(J. Chem. Phys. 120 (2004) 10240)".
            2. ref: https://sites.psu.edu/anguyennrtcapstone/example-calculation/how-to-calculate-the-d-band-center/
            3. ref: http://theory.cm.utexas.edu/forum/viewtopic.php?t=649

        Returns:
            np.float64: _description_
            
        """
        # Generate energy array for d-band centre calculation
        energy_array = np.linspace(self.energy_range[0], self.energy_range[1], self.nedos)
        ## Refer to fermi level
        energy_array -= self.fermi_level
        
        
        # Merge d-band suborbitals
        if merge_suborbitals:
            merged_d_band = np.sum(np.copy(self.d_band_array), axis=1)
        else:
            raise ValueError("Not written yet.")
        
        
        # Calculate d-band centre (referenced to fermi level)
        d_band_centre = self.__calculate_band_moment(merged_d_band, energy_array, ordinal=1)
        
        if verbose:
            print(f"d-band centre is {round(d_band_centre, 4)} eV.")
        
        
        return d_band_centre
    

# Test area
if __name__ == "__main__":
    example_dos_file = "../../0-dataset/feature_DOS/g-C3N4/3-CO_is/4-Co/dos_up_57.npy"
    d_band = dBand(dosFile=example_dos_file, fileType="numpy",
                   fermi_level_dir="../../0-dataset/z-supporting-info/fermi_level",
                   energy_range=[-14, 6],
                   )
    
    d_band_centre = d_band.calculate_d_band_centre(verbose=True)
     