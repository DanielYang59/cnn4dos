#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import os
from pathlib import Path
import re
from scipy.interpolate import interp1d
import subprocess


class Analyzer:
    def __init__(self, rootpath, atom_index):
        """DOS analyzer for DOS perturbation experiments.

        Args:
            rootpath (Path): DOS data folder.
            atom_index (int): index of atom to extract DOS from (starts from one).
            
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
    
    
    def __calculate_dos_change(self, original_dos_array, original_energy_array, perturbed_dos_array, perturbed_energy_array):
        """Calculate DOS change through interpolation.

        Args:
            original_dos_array (np.ndarray): _description_
            original_energy_array (np.ndarray): _description_
            perturbed_dos_array (np.ndarray): _description_
            perturbed_energy_array (np.ndarray): _description_
            
        Returns:
            shared_X (np.ndarray): share energy array
            difference_array (np.ndarray): DOS difference array
            
        """
        
        
        def interpolate_DOS(X, Ys, target_X):
            """Interpolate DOS for all orbital channels.

            Args:
                X (np.ndarray): energy array
                Ys (np.ndarray): DOS array in shape (NEDOS, numOrbitals)
                target_X (np.ndarray): target energy array for DOS

            Returns:
                np.ndarray: interpolated DOS array in shape (numSamples, numOrbitals)
                
            """
            # Check X and Y arrays shape
            assert X.ndim == 1 and Ys.ndim == 2
            assert target_X.ndim == 1
            assert X.shape[0] == Ys.shape[0]
            assert (target_X[0] >= X[0]) and (target_X[-1] <= X[-1])
            

            # Interpolate each DOS orbital channel
            resulted_DOSs = []
            for channel_index in range(Ys.shape[1]):            
                f1 = interp1d(X, Ys[:, channel_index])
                resulted_DOSs.append(f1(target_X))
                
            return np.swapaxes(np.stack(resulted_DOSs), 0, 1)
        
        
        # Check DOS and energy arrays
        assert original_dos_array.shape[0] == original_energy_array.shape[0] == perturbed_dos_array.shape[0] == perturbed_energy_array.shape[0]
        assert original_dos_array.shape[1] == perturbed_dos_array.shape[1]
        
        
        # Create shared energy (X) array
        x_min = max(original_energy_array[0], perturbed_energy_array[0])
        x_max = min(original_energy_array[-1], perturbed_energy_array[-1])
        step_length = (self.energy_range[1] - self.energy_range[0]) / self.energy_range[2]
        shared_X = np.arange(x_min, x_max, step_length)  # NOTE: did not add step length
        
        
        # Interpolate DOS (Y) arrays
        interpolated_original_array = interpolate_DOS(X=original_energy_array, Ys=original_dos_array, target_X=shared_X)
        interpolated_perturbed_array = interpolate_DOS(X=perturbed_energy_array, Ys=perturbed_dos_array, target_X=shared_X)
        
        
        # Calculate DOS difference
        difference_array = interpolated_perturbed_array - interpolated_original_array
        
        return shared_X, difference_array
        
        
    def __extract_dos_for_dir(self, path, dos_script, single_dos_script):
        """DOS extraction worker for a folder.

        Args:
            path (Path): DOS dir to extract single atom DOS.
            atom_index (int): index of atom to extract DOS.
            dos_script (Path): extract DOS from vasprun.xml script.
            single_dos_script (Path): extract single atom DOS script.
            
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
        
        
        assert dos_array.shape[0] == energy_array.shape[0]
        return dos_array, energy_array
    
    
    def __write_dos_change(self, path, energy_array, dos_change_array):
        """Write DOS change array and energy array to file.

        Args:
            path (Path): target directory
            energy_array (np.ndarray): energy array for DOS change
            dos_change_array (np.ndarray): DOS change array
            
        """
        # Create target directory
        os.makedirs(path, exist_ok=True)


        # Write energy array
        np.save(path / "energy.npy", energy_array)
        np.save(path / "dos_change.npy", dos_change_array)

    
    def calculate_dos_change(self, energy_range):
        # Check energy_range (roughly)
        assert len(energy_range) == 3 and isinstance(energy_range, tuple)
        self.energy_range = energy_range
        
        
        # Get original DOS and energy arrays
        original_dos_array, original_energy_array = self.__get_dos(self.original_path)
        
        
        # For each folder under "path", calculate DOS change
        for folder in self.perturbed_path.iterdir():
            if folder.is_dir() and not folder.name.startswith("."):
                # Get perturbed DOS and energy arrays
                perturbed_dos_array, perturbed_energy_array = self.__get_dos(folder)
                
                
                # Calculate DOS change
                dos_change = self.__calculate_dos_change(
                    original_dos_array, original_energy_array,
                    perturbed_dos_array, perturbed_energy_array,
                    )
                
                
                # Write results to local file
                source_path = list(folder.resolve().parts)
                source_path[-4] = "results"  # NOTE: might use other names
                output_path = Path(*source_path)
                
                self.__write_dos_change(path=output_path, energy_array=dos_change[0], dos_change_array=dos_change[1])
                

    def extract_dos(self, dos_script, single_dos_script, verbose=True):
        """Extract DOS from vasprun.xml for perturbation analysis.

        Args:
            dos_script (Path): extract DOS from vasprun.xml script.
            single_dos_script (Path): extract single atom DOS script.
            verbose (bool): verbose during extraction.
            
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
    
    
    # Test DOS extraction
    analyzer.extract_dos(
                         Path("../../../0-utils/extract_dos_from_vasprunxml.py").resolve(),
                         Path("../../../0-utils/extract_single_atom_DOS.py").resolve()
                         )
    
    
    # Test DOS change calculation
    analyzer.calculate_dos_change(energy_range=(-14, 6, 4000))
