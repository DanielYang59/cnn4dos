"""Load, manipulate and save electronic density of state (eDOS)
data, for VASP jobs.

# TODO:
- Check eDOS shape when loading from local numpy array file and vasprun
- Finish preprocess method
- Avoid hard coding in remove_ghost_state (energy axis selection)
- Make sure no copies are made during "remove_ghost_states",
    "reset_axes" and "preprocess".
"""

import itertools
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from pymatgen.io.vasp import Vasprun

from cnn4dos.data.core import Orbitals, Spins


class Edos:
    """Handle eDOS as numpy array, default in
        shape ("orbital", "energy", "atom", "spin").

    Parameters:
    - edos_arr (Optional[np.ndarray], optional): The eDOS array.
        Defaults to None, and could be parsed later.
    - expected_shape (Optional[tuple[int, int, int]], optional): The expected
        shape of the eDOS array. Defaults to None.
    - axes (tuple[str, str, str], optional): The axes labels for the
        eDOS array. Defaults to ("orbital", "energy", "atom", "spin").

    Attributes:
    - edos_arr: The 4D eDOS numpy array (expand_dims if not).
    - expected_shape: The expected shape of the eDOS array.
    - axes: The axes labels for the eDOS array.

    Raises:
    - ValueError: If the provided `expected_shape` is not a 4-tuple,
        or if the provided `axes` are not a tuple with
        elements 'orbital', 'energy', 'atom' and 'spin'.
    """

    def __init__(
        self,
        edos_arr: Optional[np.ndarray] = None,
        expected_shape: Optional[tuple[int, int, int, int]] = None,
        axes: tuple[str, str, str, str] = (
            "orbital",
            "energy",
            "atom",
            "spin",
        ),
    ) -> None:
        # Check arg: edos_arr
        if edos_arr is not None:
            if not isinstance(edos_arr, np.ndarray) or edos_arr.ndim != 4:
                raise ValueError("Expect a 4D eDOS numpy array.")

        # Check arg: expected_shape
        if expected_shape is not None and len(expected_shape) != 4:
            raise ValueError("Expect a 4D shape.")

        # Check arg: axes
        if axes is not None and (
            not isinstance(axes, tuple)
            or len(axes) != 4
            or sorted(axes) != sorted(("orbital", "energy", "atom", "spin"))
        ):
            raise ValueError(
                "Axes should be tuple with only orbital, energy, atom, spin."
            )

        self.edos_arr = edos_arr
        self.axes = axes
        self.expected_shape = expected_shape

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

    def remove_ghost_states(self, e_axis: int, width: int = 1) -> None:
        """Remove ghost states from eDOS array, along energy axis.

        During eDOS calculations, we observed that in some cases,
        there is a spike occurring exclusively at the 0th point of
        the entire eDOS along energy axis. We verified the unreality of such
        spike by sliding the eDOS energy windows; while the original spike
        disappears, a new spike emerges at the new 0th position.

        Although the exact cause of this phenomenon remains unknown,
        we decided to remove it anyway. Because its presence would yield data
        preprocessing tricky such as normalization and standardization.

        Parameters:
        - e_axis (int): The index of the energy axis in the eDOS array.
        - width (int, optional): The width along energy axis to remove
            ghost states. Defaults to 1.

        Raises:
        - ValueError: If the provided axis index is invalid or if the removing
            width is not a positive integer or exceeds the total eDOS width.
        """

        # Check arg: width
        if not isinstance(width, int) or width <= 0:
            raise ValueError("Removing width should be a positive integer.")
        elif width > self.array.shape[e_axis]:
            raise ValueError("Removing width greater than total width.")

        # Remove ghost states by setting corresponding values to zero
        # TODO: avoid hard coding
        if e_axis == 0:
            self.array[:width, :, :] = 0.0
        elif e_axis == 1:
            self.array[:, :width, :] = 0.0
        elif e_axis == 2:
            self.array[:, :, :width] = 0.0
        else:
            raise ValueError("Axis index out of range.")

    def reset_axes(self, new_axes: tuple[str, str, str, str]) -> None:
        """Reset the axes of the eDOS array based on new axes labels.
            Axes should be tuple with only "orbital",
            "energy", "atom" and "spin".

        Parameters:
            new_axes (tuple[str, str, str, str]): The new axes labels.

        Returns:
            None
        """
        # Check new axes
        if not (
            isinstance(new_axes, tuple)
            and len(new_axes) == 4
            and sorted(new_axes)
            == sorted(("orbital", "energy", "atom", "spin"))
        ):
            raise ValueError(
                "Axes should be tuple with only orbital, energy, atom, spin."
            )

        # Reset axes by swapping axies
        if self.edos_arr is None:
            raise ValueError("Cannot reset axes where no edos_arr is set.")

        elif new_axes == self.axes:
            warnings.warn("Old and new axes are the same.")

        else:
            # Get the permutation required to swap axes based on the new order
            axis_map = {axis: idx for idx, axis in enumerate(self.axes)}
            permutation = [axis_map[axis] for axis in new_axes]

            # Apply the permutation
            self.axes = new_axes
            self.edos_arr = np.transpose(self.edos_arr, permutation)

    def to_array(self, filename: Path) -> None:
        """Save eDOS array to a numpy array file (.npy).

        Parameters:
            filename (Path): The path to save the numpy array file.

        Returns:
            None
        """

        np.save(filename, self.array)

    # def preprocess(self, method: Union[Callable, str], axis: int) -> None:
    #     """Preprocess eDOS array.
    #     TODO:

    #     """

    #     if method in {"normalize", "normalise"}:
    #         pass

    #     elif method in {"standardize", "standardise"}:
    #         pass

    #     # elif isinstance(method, Callable):
    #     #     pass

    #     else:
    #         raise ValueError(f"Unsupported preprocess method {method}.")