#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOS extraction script for CNN training.
Extract PDOS of ALL atoms from vasprun.xml.
"""


spin = "up"


import os
import sys
import numpy as np


def get_total_atom(poscarfile):
    """Get total number of atoms from POSCAR or CONTCAR.

    Args:
        poscarfile (str): target POSCAR or CONTCAR file.

    Returns:
        int: total number of atoms
        
    """
    # Check filename
    assert poscarfile.split(os.sep)[-1] in {"POSCAR", "CONTCAR"}
    
    # Import POSCAR or CONTCAR file
    with open(poscarfile) as f:
        poscar_data = f.readlines()

    # Get total number of atoms
    atom_num_line = [int(i) for i in poscar_data[6].strip().split()]

    return sum(atom_num_line)


def dos_extractor(folder, spin):
    """Extractor DOS from vasprun.xml from given folder.

    Args:
        folder (str): name of folder to work on.
        spin (str): extract spin "up", "down" or "both".
        
    """
    
    def read_ispin_from_vaspxml(vasprunxml_root):
        """Read ISPIN tag value from vasprun.xml.

        Returns:
            str: ISPIN tag in value either "1" or "2"
            
        """
        # Get ISPIN value
        index = 0
        for child in dos_root[1]:
            # locate ISPIN tag position
            if dos_root[1][index].attrib["name"] == "ISPIN":
                break
            else:
                index += 1

        ispin = dos_root[1][index].text.strip()

        assert ispin in {"1", "2"}
        return ispin

    # Check args
    assert spin in {"up", "down", "both"}
    assert os.path.exists(os.path.join(folder, "vasprun.xml"))
    
    # Get vasprun.xml filename
    file = os.path.join(folder, "vasprun.xml")

    # Import vasprun.xml
    from xml.etree import ElementTree
    dos_tree = ElementTree.parse(file)
    dos_root = dos_tree.getroot()

    # Decide if spin polarized from "ISPIN" tag
    ispin = read_ispin_from_vaspxml(dos_root)

    # Begin to export DOS data from vasprun.xml
    if ispin == "1":
        return ValueError("None spin polarised analysis currently not supported!")

    else:  # ISPIN = 2
        # create empty list
        dos_up = []
        dos_down = []

        # find dos data position
        for atom in dos_root[8][-2][-1][0][-1]:  # 8-calculation; -2-dos; -1-partial; 0-array; -1-set
            # Spin up
            spin_up_dos = [i.text.strip() for i in atom[0]]
            ## Convert each line to np array
            spin_up_dos = np.array([[float(j) for j in i.split()] for i in spin_up_dos])

            # Spin down
            spin_down_dos = [i.text.strip() for i in atom[1]]
            ## Convert each line to np array
            spin_down_dos = np.array([[float(j) for j in i.split()] for i in spin_down_dos])

            # Append DOS data to list
            dos_up.append(spin_up_dos)
            dos_down.append(spin_down_dos)

        # Stack dos arrays
        dos_up = np.array(dos_up)
        dos_down = np.array(dos_down)

    # Write DOS to numpy array
    if ispin == "1":
        return ValueError("None spin polarised analysis currently not supported!")  # DEBUG

    else:  # ISPIN = 2
        if spin in {"up", "both"}:
            # write spin up as "dos_up.npy"
            np.save(os.path.join(folder, "dos_up.npy"), dos_up)
        if spin in {"down", "both"}:
            # write spin down as "dos_down.npy"
            np.save(os.path.join(folder, "dos_down.npy"), dos_down)


def show_progress_bar(index, total, label):
    n_bar = 50  # Progress bar width
    progress = index / total
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * progress):{n_bar}s}] {int(100 * progress)}%  {label}")
    sys.stdout.flush()


def get_folder_list(path):
    folder_list = []  # a list of subfolders
    for folder in os.listdir(path):
        if os.path.isdir(f"{path}/{folder}") and not folder.startswith("."):
            folder_list.append(folder)
    folder_list.sort()  # sort subfolder list

    return folder_list


if __name__ == "__main__":
    # Verbose
    print(f"Extracting all atoms PDOS, spin {spin}.")
    
    # Get folder list
    folder_list = get_folder_list(path=".") 
    total_folder_num = len(folder_list)

    # For each folder
    for index, folder in enumerate(folder_list):
        # decide if essential file exists (POSCAR and vasprun.xml)
        if os.path.exists(os.path.join(folder, "POSCAR")) and os.path.exists(os.path.join(folder, "vasprun.xml")):
            # Get total number of atoms from POSCAR
            atom_num = get_total_atom(os.path.join(folder, "POSCAR"))

            # Show progress bar
            show_progress_bar(index + 1, total_folder_num, f"Extracting {index + 1} out of {total_folder_num}")

            # Run DOS extractor
            dos_extractor(folder, spin=spin)

        else:
            print(f"POSCAR or vasprun.xml not found in \"{folder}\"")

    # All jobs done
    print("\nAll done.")