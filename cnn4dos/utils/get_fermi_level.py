"""Get fermi level for a VASP run."""

from pathlib import Path

from pymatgen.io.vasp import Outcar, Vasprun, Wavecar


def get_fermi_level(path: Path) -> float:
    """Get the Fermi level from VASP output files.

    This function searches for the Fermi level in VASP output files
    including vasprun.xml, OUTCAR, and WAVECAR.

    Parameters:
        path (Path): The directory path to the VASP task.

    Returns:
        float: The Fermi level.

    Raises:
        FileNotFoundError: If no file containing the Fermi level is found.
    """
    # Compile possible files to search
    vasprun_file = path / "vasprun.xml"
    outcar_file = path / "OUTCAR"
    wavecar_file = path / "WAVECAR"

    if vasprun_file.is_file():
        return Vasprun(vasprun_file).efermi
    elif outcar_file.is_file():
        return Outcar(outcar_file).efermi
    elif wavecar_file.is_file():
        return Wavecar(wavecar_file).efermi

    else:
        raise FileNotFoundError("Cannot find any file containing fermi level.")
