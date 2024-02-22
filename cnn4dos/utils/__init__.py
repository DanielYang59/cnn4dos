"""Utility scripts and globally useful variables."""

from pathlib import Path

# Local imports
from .convert_atom_selection import convert_atom_selection  # noqa: F401
from .get_fermi_level import get_fermi_level  # noqa: F401
from .list_folders import list_folders  # noqa: F401

# Root directory of the package
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
