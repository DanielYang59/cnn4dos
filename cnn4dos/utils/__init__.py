"""Utility scripts and globally useful variables."""


from pathlib import Path

from pymatgen.electronic_structure.core import Orbital, Spin

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
    "dx2": Orbital(8)
}

Spins = {
    "up": Spin.up,
    "down": Spin.down
}
