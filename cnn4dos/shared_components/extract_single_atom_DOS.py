"""Extract pDOS from the numpy array of a selected atom."""

spin = "up"
working_dir = "."


import os
from pathlib import Path

import numpy as np


def extract_DOS(source_dir, atom_index, spin="up"):
    # Check args
    assert spin in {"up", "down", "both"}
    assert isinstance(atom_index, int) and atom_index >= 1

    # Load spin up DOS
    if spin in {"up", "both"}:
        # Load source eDOS array
        source_arr = np.load(source_dir / "dos_up.npy")

        # Extract selected atom eDOS (spd only)
        target_arr = source_arr[atom_index - 1, :, 1:10]

        # Write new eDOS array
        np.save((source_dir / f"dos_up_{atom_index}.npy"), target_arr)

    # Load spin down DOS
    if spin in {"down", "both"}:
        # Load source eDOS array
        source_arr = np.load(source_dir / "dos_down.npy")

        # Extract selected atom eDOS (spd only)
        target_arr = source_arr[atom_index - 1, :, 1:10]

        # Write new eDOS array
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
        extract_DOS(source_dir=working_dir, atom_index=atom_index, spin=spin)
