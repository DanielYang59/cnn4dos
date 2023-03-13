#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import glob
import os


def list_dos_files(dos_dir, adsorbates, substrates, dos_file_name="dos_up_*.npy"):
    """List all DOS files based on adsorbate and substrate name.

    Args:
        dos_dir (str): DOS file directory
        adsorbates (list): list of adsorbates to search
        substrates (list): list of substrates to search
        dos_file_name (str, optional): DOS file name pattern. Defaults to "dos_up_*.npy".

    Raises:
        ValueError: if DOS filename pattern has no match

    Returns:
        list: all matched DOS files
        
    """
    # Check args
    assert os.path.isdir(dos_dir)
    assert isinstance(adsorbates, list)
    assert isinstance(substrates, list)
    
    
    # Get all DOS files
    dos_files = []
    for sub in substrates:
        for ads in adsorbates:
            dos_file_pattern = os.path.join(dos_dir, sub, ads, "*", dos_file_name)
            matches = glob.glob(dos_file_pattern)
            if not matches:
                raise ValueError(f"No matched DOS found for pattern {dos_file_pattern}.")
            dos_files.extend(matches)
    
    
    return dos_files
    

# Test area
if __name__ == "__main__":
    
    dos_files = list_dos_files(
        dos_dir="../../0-dataset/feature_DOS",
        adsorbates=["3-CO_is", ],
        substrates = ["g-C3N4", "nitrogen-graphene", "vacant-graphene", "C2N", "BN", "BP"],
    )
    print(dos_files)
    