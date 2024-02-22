"""Load, manipulate and save electronic density of state (eDOS)
data, for VASP jobs.
"""

# TODOs:
# finish remove ghost state and preprocess methods

import itertools
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from pymatgen.io.vasp import Vasprun

from cnn4dos.utils import Orbitals, Spins


class eDOS:
    """Handle eletronic density of states (eDOS) as numpy array."""

    def __init__(self, shape: Optional[tuple[int]] = None) -> None:
        self.expected_shape = shape

    def _check_shape(self) -> bool:
        """Check if the shape of the eDOS array matches the expected shape.

        Returns:
            bool: Whether the shape matches the expected shape.

        Notes:
            If the expected shape is not set, a warning is issued,
                and the function returns False.
        """

        if self.expected_shape is None:
            warnings.warn("Cannot check eDOS shape without expected shape.")
            return False

        elif self.array.shape == self.expected_shape:
            return True

        else:
            return False

    def from_array(self, filename: Path) -> None:
        """Load eDOS directly from numpy array file.

        Parameters:
            filename (Path): The path to the numpy array file.

        Returns:
            None

        Raises:
            Warning: If the file extension is not '.npy'.
        """

        # Check file extension
        if filename.suffix != ".npy":
            warnings.warn("eDOS file extension is not .npy.")

        # Load eDOS array
        self.array = np.load(filename)

    def from_vasprun(
        self,
        filename: Path,
        atoms: list[int],
        orbitals: list[str],
        spins: list[str] = ["up", "down"],
    ) -> None:
        """Load eDOS from vasprun.xml for a VASP run.

        Parameters:
            filename (Path): Path to the vasprun.xml file.
            atoms (list[int]): The list of atom indices.
            orbitals (list[str]): The list of orbital strings.
            spins (list[str], optional): The list of spin strings.
                Defaults to ["up", "down"].

        Notes:
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
            isinstance(atom, int) and atom in range(total_atoms)
            for atom in atoms
        ):
            raise ValueError(
                f"atom index must be int in range [0, {total_atoms - 1}]",
            )
        if len(set(atoms)) != len(atoms):
            raise ValueError("Duplicate atoms not allowed.")

        if not all(
            isinstance(orb, str) and orb in Orbitals for orb in orbitals
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

        self.array = np.array(results).reshape(
            len(atoms), len(orbitals), len(spins), *results[0].shape
        )

    def preprocess(self, method: str, axis: int) -> None:
        # TODO:

        if method == "normalize":
            pass

        elif method == "standardize":
            pass

        else:
            raise ValueError(f"Unsupported proprocess method {method}.")

    def remove_ghost_states(self, axis: int, indexes: list[int]) -> None:
        """Remove ghost states from eDOS.

        During the calculation of eDOS, we observed that in certain cases,
        there is a significant spike occurring exclusively at the 0th point of
        the entire eDOS. We have verified the unreality of this spike by
        adjusting the energy windows; while the original spike disappears,
        a new spike emerges at the new 0th position.

        Although the exact cause of this phenomenon remains unknown,
        we have decided to remove it. Its presence would complicate data
        preprocessing tasks such as normalization and standardization.

        """
        # Check axis
        if not (
            isinstance(axis, int) and axis in range(len(self.array.shape))
        ):
            raise ValueError(f"Invalid axis {axis}.")

        # Check indexes
        # TODO:

        # Remove ghost states by setting corresponding values to zero
        # self.array[0, :] = 0.0

    def swap_axes(self, axis1: int, axis2: int) -> None:
        """Swap the axes of the eDOS array.

        Parameters:
            axis1 (int): The first axis to be swapped.
            axis2 (int): The second axis to be swapped.

        Returns:
            None

        Notes:
            For example, if you have an array with shape (a, b),
            calling swap_axes(arr, 0, 1) will result in the array's
            shape being transformed to (b, a).
        """

        self.array = np.swapaxes(self.array, axis1, axis2)

    def to_array(self, filename: Path) -> None:
        """Save eDOS array to a numpy array file (.npy).

        Parameters:
            filename (Path): The path to save the numpy array file.

        Returns:
            None
        """

        np.save(filename, self.array)
