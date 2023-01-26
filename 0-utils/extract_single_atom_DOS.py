#!/usr/bin/env python3
# -*- coding: utf-8 -*-


spin = "up"


import os
import numpy as np


def fetch_dos(source_dir, atom_index, spin="up", remove_ghost=False):
    # Check args
    assert spin in {"up", "down", "both"}
    assert isinstance(atom_index, int) and atom_index >= 1
    
    
    # Load spin up DOS
    if spin in {"up", "both"}:
        # Load source DOS array
        source_arr = np.load(os.path.join(source_dir, "dos_up.npy"))
        
        # Extract selected atom DOS (spd only)
        target_arr = source_arr[atom_index - 1, :, 1:10]
        
        # Write new DOS array
        np.save(os.path.join(source_dir, f"dos_up_{atom_index}.npy"), target_arr)
        
    # Load spin down DOS
    if spin in {"down", "both"}:
        # Load source DOS array
        source_arr = np.load(os.path.join(source_dir, "dos_down.npy"))
        
        # Extract selected atom DOS (spd only)
        target_arr = source_arr[atom_index - 1, :, 1:10]
        
        # Zero out first point to remove "ghost state"
        if remove_ghost:
            target_arr[0] = 0
        
        # Write new DOS array
        np.save(os.path.join(source_dir, f"dos_down_{atom_index}.npy"), target_arr)
        

if __name__ == "__main__":
    # Check args
    working_dir = "."
    assert os.path.isdir(working_dir)
    
    # Get atom index from user (starts from 1)
    atom_index = int(input("Which atom to extract (starts from 1)?"))
    
    # Loop through folders in working dir
    count = 0
    for d in os.listdir(working_dir):
        if os.path.exists(os.path.join(working_dir, d, "dos_up.npy")) or os.path.exists(os.path.join(working_dir, d, "dos_down.npy")):
            # Verbose
            count += 1
            print(f"Working on atom index {atom_index} spin {spin}, number {count}")
            
            # Fetch DOS
            fetch_dos(source_dir=os.path.join(working_dir, d), 
                      atom_index=atom_index, spin=spin)
