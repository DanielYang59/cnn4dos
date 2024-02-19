"""Get fermi level for a VASP run."""

from pathlib import Path

from pymatgen.io.vasp import Outcar, Vasprun, Wavecar


def get_fermi_level(path: Path, digits: int = 4) -> float:
    """Get the Fermi level from VASP output files.

    This function searches for the Fermi level in VASP output files
    including vasprun.xml, OUTCAR, and WAVECAR.

    Parameters:
        path (Path): The directory path to the VASP task.
        digits (Optional[int]): The number of digits to keep after
            the decimal point. Defaults to 4.

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
        fermi_level = Vasprun(vasprun_file).efermi
    elif outcar_file.is_file():
        fermi_level = Outcar(outcar_file).efermi
    elif wavecar_file.is_file():
        fermi_level = Wavecar(wavecar_file).efermi

    else:
        raise FileNotFoundError("Cannot find any file containing fermi level.")

    return round(fermi_level, digits)
