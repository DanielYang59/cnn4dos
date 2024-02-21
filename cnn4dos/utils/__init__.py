"""Utility scripts and globally useful variables."""

from pathlib import Path

from pymatgen.electronic_structure.core import Orbital, Spin

# Local imports
from .convert_atom_selection import convert_atom_selection  # noqa: F401
from .get_fermi_level import get_fermi_level  # noqa: F401
from .list_folders import list_folders  # noqa: F401

# Root directory of the package
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Define orbitals and spins globally
Orbitals = {
    "s": Orbital(0),
    "py": Orbital(1),
    "pz": Orbital(2),
    "px": Orbital(3),
    "dxy": Orbital(4),
    "dyz": Orbital(5),
    "dz2": Orbital(6),
    "dxz": Orbital(7),
    "dx2": Orbital(8),
}

Spins = {"up": Spin.up, "down": Spin.down}
