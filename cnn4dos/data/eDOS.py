"""Utilities for load, manipulate and save electronic density of state (eDOS)
data, for VASP jobs.
"""

import itertools
from pathlib import Path
import numpy as np
import warnings

from pymatgen.io.vasp import Vasprun

from cnn4dos.utils import Orbitals, Spins


class eDOS:
    """Handle eletronic density of states (eDOS) as numpy array.
    """

    def __init__(self, shape: tuple[int] = None) -> None:
        pass

    def from_vasprun(
        self,
        filename: Path,
        atoms: list[int],
        orbitals: list[str],
        spins: list[str] = ["up", "down"]
    ) -> np.ndarray:
        """Load eDOS from vasprun.xml for a VASP run.

        Parameters:
            filename (Path): Path to the vasprun.xml file.
            atoms (list[int]): The list of atom indices.
            orbitals (list[str]): The list of orbital strings.
            spins (list[str], optional): The list of spin strings.
                Defaults to ["up", "down"].

        Returns:
            np.ndarray: The eDOS array in shape:
                (len(atoms), len(orbitals), len(spins), NEDOS), where NEDOS
                is the VASP INCAR tag (number of grid points).

        Example:
            >>> edos = eDOS()
            >>> edos.from_vasprun(
            ...     filename="vasprun.xml",
            ...     atoms=[0, 1, 2],
            ...     orbitals=["s", "py"],
            ...     spins=["up", "down"]
            ... )
        """

        # Load vasprun.xml
        vasprun = Vasprun(filename=filename)

        # Check arguments
        total_atoms = len(vasprun.atomic_symbols)

        if not all(
            isinstance(atom, int)
            and atom in range(total_atoms) for atom in atoms
                ):
            raise ValueError(
                f"atom index must be int in range [0, {total_atoms - 1}]"
            )
        if len(set(atoms)) != len(atoms):
            raise ValueError("Duplicate atoms not allowed.")

        if not all(
            isinstance(orb, str)
            and orb in Orbitals for orb in orbitals
                ):
            raise ValueError("Invalid orbital found.")
        if len(set(orbitals)) != len(orbitals):
            raise ValueError("Duplicate orbitals not allowed.")

        if not all(spin in Spins for spin in spins):
            raise ValueError("spin must be either 'up' or 'down'.")
        if len(set(spins)) != len(spins):
            raise ValueError("Duplicate spins not allowed.")

        # Parse pDOS
        pdos = vasprun.pdos

        results = []
        for atom, orb, spin in itertools.product(atoms, orbitals, spins):
            # pDOS stored as pdos[atomindex][orbital][spin]
            arr = pdos[atom][Orbitals[orb]][Spins[spin]]
            results.append(arr)

        return np.array(results).reshape(
            len(atoms), len(orbitals), len(spins), *results[0].shape
        )

    def from_array(self, filename: Path) -> np.ndarray:
        """Load eDOS directly from numpy array file.

        Parameters:
            filename (Path): The path to the numpy array file.

        Returns:
            np.ndarray: The eDOS array loaded from the numpy array file.

        Raises:
            Warning: If the file extension is not '.npy'.
        """

        # Check file extension
        if filename.suffix != ".npy":
            warnings.warn("eDOS file extension is not .npy.")

        # Load eDOS array
        return np.load(filename)

    def to_array(self):
        pass

    def remove_ghost_state(self):
        pass

    def swap_axes(self):
        pass

    def check_shape(self):
        pass
