#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import os
import re
import subprocess


class Analyzer:
    def __init__(self, rootpath, atom_index) -> None:
        """DOS analyzer for DOS perturbation experiments.

        Args:
            rootpath (Path): DOS data folder.
            atom_index (int): index of atom to extract DOS from (starts from one)
            
        """
        # Check working directory
        assert rootpath.is_dir()
        assert (rootpath / "0-original").exists()
        assert (rootpath / "1-perturbed").exists()
        self.rootpath = rootpath
        self.original_path = rootpath / "0-original"
        self.perturbed_path = rootpath / "1-perturbed"
        
        
        # Check atom index
        assert isinstance(atom_index, int) and atom_index >= 1
        self.atom_index = atom_index
    
    
    def __extract_dos_for_dir(self, path, dos_script, single_dos_script):
        """DOS extraction worker for a folder.

        Args:
            path (Path): DOS dir to extract single atom DOS
            atom_index (int): index of atom to extract DOS
            dos_script (Path): extract DOS from vasprun.xml script
            single_dos_script (Path): extract single atom DOS script
            
        """
        # Check path
        assert path.exists()
        assert dos_script.exists()
        assert single_dos_script.exists()
        
        
        # Work on target directory
        pwd = os.getcwd()
        os.chdir(path)
        
        ## Run extract DOS from vasprun.xml script
        subprocess.run(["python3", dos_script])
        
        ## Run extract single atom DOS script
        p = subprocess.Popen(["python3", single_dos_script], 
                             stdin=subprocess.PIPE, 
                             stdout=subprocess.DEVNULL)
        p.communicate(input=f"{self.atom_index}".encode())

        
        # Get back to original directory
        os.chdir(pwd)
    
    
    def __get_dos(self, path, spin="up"):
        """Get DOS and energy array (referred to fermi level) from selected project folder.

        Args:
            path (Path): folder to get DOS and energy arrays from.
            spin (str, optional): spin channel. Defaults to "up".

        Returns:
            tuple: (DOS_array, energy_array)
            
        """
        # Check path
        assert path.exists()
        
        
        # Get fermi level
        assert (path / "vasprun.xml").exists()
        output = subprocess.getoutput([f"grep fermi {path / 'vasprun.xml'}"])
        fermi_level = float(re.findall('<i name="efermi">\s+(-?\d+\.\d+)\s+</i>', output)[0])
       
        
        # Get DOS array in shape (NEDOS, numOrbitals)
        assert (path / f"dos_{spin}_{self.atom_index}.npy").exists()
        dos_array = np.load(path / f"dos_{spin}_{self.atom_index}.npy")
        
        
        # Generate energy array (for DOS)
        energy_array = np.linspace(self.energy_range[0], self.energy_range[1], self.energy_range[2])
        energy_array -= fermi_level
        
        
        return dos_array, energy_array        
        
        
    def calculate_dos_change(self, energy_range):
        # Check energy_range (roughly)
        assert len(energy_range) == 3 and isinstance(energy_range, tuple)
        self.energy_range = energy_range
        
        
        # Get original DOS
        self.__get_dos(self.original_path)
        
        
        # For each folder under "path", calculate DOS change
        for d in self.perturbed_path.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                # print(d) #DEBUG
                pass
                

    def extract_dos(self, dos_script, single_dos_script, verbose=True):
        """Extract DOS from vasprun.xml for perturbation analysis.

        Args:
            dos_script (Path): extract DOS from vasprun.xml script
            single_dos_script (Path): extract single atom DOS script
            verbose (bool): verbose during extraction
            
        """
        # Work on "0-original" dir
        self.__extract_dos_for_dir(self.original_path, dos_script, single_dos_script)
        
        
        # List all folders
        directories = []
        for d in (self.perturbed_path.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                directories.append(d)
        
        
        # Work on each subdir of "1-perturbed" dir 
        for i, d in enumerate(directories):
            self.__extract_dos_for_dir(d, dos_script, single_dos_script)
            
            if verbose:
                print(f"DOS extraction on {i + 1} out of {len(directories)} folders.")
    

# Test area
if __name__ == "__main__":
    from pathlib import Path
    
    # Initiate test project
    atom_index = 71
    analyzer = Analyzer(
        rootpath=Path("../data") / "1-doping",
        atom_index=atom_index
        )
    
    
    # # Test DOS extraction
    # analyzer.extract_dos(
    #                      Path("../../../0-utils/extract_dos_from_vasprunxml.py").resolve(),
    #                      Path("../../../0-utils/extract_single_atom_DOS.py").resolve(),
    #                      )
    
    
    # Test DOS change calculation
    analyzer.calculate_dos_change(energy_range=(-14, 6, 4000))
