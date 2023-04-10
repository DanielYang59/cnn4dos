#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import subprocess


class Analyzer:
    def __init__(self, path) -> None:
        """DOS analyzer for DOS perturbation experiments.

        Args:
            path (Path): DOS data folder.
            
        """
        # Check working directory
        assert path.is_dir()
        assert (path / "0-original").exists()
        assert (path / "1-perturbed").exists()
        self.path = path

    
    def __extract_dos_for_dir(self, path, atom_index, dos_script, single_dos_script):
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
        p.communicate(input=f"{atom_index}".encode())

        os.chdir(pwd)
        
    
    def extract_dos(self, atom_index, dos_script, single_dos_script, verbose=True):
        """Extract DOS from vasprun.xml for perturbation analysis.

        Args:
            atom_index (int): index of atom to extract DOS from (starts from one)
            dos_script (Path): extract DOS from vasprun.xml script
            single_dos_script (Path): extract single atom DOS script
            verbose (bool): verbose during extraction
            
        """
        # Check args
        assert isinstance(atom_index, int) and atom_index >= 1
        
        
        # Work on "0-original" dir
        self.__extract_dos_for_dir((self.path / "0-original"), atom_index, dos_script, single_dos_script)
        
        
        # List all folders
        directories = []
        for d in (self.path / "1-perturbed").iterdir():
            if d.is_dir() and not d.name.startswith("."):
                directories.append(d)
        
        
        # Work on each subdir of "1-perturbed" dir 
        for i, d in enumerate(directories):
            self.__extract_dos_for_dir(d, atom_index, dos_script, single_dos_script)
            
            if verbose:
                print(f"DOS extraction on {i + 1} out of {len(directories)} folders.")
        

# Test area
if __name__ == "__main__":
    from pathlib import Path
    # Work on selected project
    analyzer = Analyzer(Path("../data") / "1-doping")
    
    
    # Extract DOS of selected atom
    atom_index = 66
    analyzer.extract_dos(atom_index, 
                         Path("../../../0-utils/extract_dos_from_vasprunxml.py").resolve(),
                         Path("../../../0-utils/extract_single_atom_DOS.py").resolve(),
                         )
