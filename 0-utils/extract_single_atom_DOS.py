#!/usr/bin/env python3
# -*- coding: utf-8 -*-


spin = "up"
working_dir = "."


import numpy as np
import os
from pathlib import Path


def extract_DOS(source_dir, atom_index, spin="up"):
    # Check args
    assert spin in {"up", "down", "both"}
    assert isinstance(atom_index, int) and atom_index >= 1
    
    
    # Load spin up DOS
    if spin in {"up", "both"}:
        # Load source DOS array
        source_arr = np.load(source_dir / "dos_up.npy")
        
        # Extract selected atom DOS (spd only)
        target_arr = source_arr[atom_index - 1, :, 1:10]
        
        # Write new DOS array
        np.save((source_dir / f"dos_up_{atom_index}.npy"), target_arr)
        
    # Load spin down DOS
    if spin in {"down", "both"}:
        # Load source DOS array
        source_arr = np.load(source_dir / "dos_down.npy")
        
        # Extract selected atom DOS (spd only)
        target_arr = source_arr[atom_index - 1, :, 1:10]
        
        # Write new DOS array
        np.save((source_dir / f"dos_down_{atom_index}.npy"), target_arr)
        

if __name__ == "__main__":
    # Check args
    working_dir = Path(working_dir)
    assert os.path.isdir(working_dir)
    
    # Get atom index from user (starts from 1)
    atom_index = int(input("Which atom to extract (starts from 1)?"))
    
    # Loop through folders in working dir
    if (working_dir / f"dos_{spin}.npy").exists():
        # Extract DOS
        extract_DOS(source_dir=working_dir, 
                      atom_index=atom_index, spin=spin)
