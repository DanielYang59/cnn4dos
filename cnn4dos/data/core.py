"""Global definitions."""

from pymatgen.electronic_structure.core import Orbital, Spin

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