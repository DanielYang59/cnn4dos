# Configs
dos_filename = "dos_up_71.npy"

working_dir = "."
original_DOS_dir = "/Users/yang/Library/CloudStorage/OneDrive-Personal/0-QUT-ProjectData/3-DOS神经网络/4-doping-experiments/calculate-DOS-difference/0-original-Cr-graphene"


import re
import numpy as np
from pathlib import Path
import subprocess


def calculate_dos_difference(dos_array, ref_dos_array, result_array_path):
    # Check args
    assert isinstance(dos_array, np.ndarray) and isinstance(ref_dos_array, np.ndarray)
    assert dos_array.shape == ref_dos_array.shape


    # Calculate difference array and save to file
    diff_dos_array =  dos_array - ref_dos_array

    np.save(
        result_array_path,
        diff_dos_array
            )


def get_fermi_level(dir):
    """Get fermi level for vasprun.xml file.

    Args:
        dir (Path): _description_

    Returns:
        float: _description_
    """
    # Check vasprun.xml path
    p = Path(dir) / "vasprun.xml"
    assert p.is_file()
    
    # Get fermi level
    output = subprocess.getoutput(f'grep fermi {Path(dir) / "vasprun.xml"}')
    
    return float(re.findall('<i name="efermi">\s+(-?\d+\.\d+)\s+</i>', output)[0])


if __name__ == "__main__":
    # Load DOS array and reference DOS array
    dos_array = np.load(Path(working_dir) / dos_filename)
    ref_dos_array = np.load(Path(original_DOS_dir) / dos_filename)
    
    
    # Compare two DOSs
    calculate_dos_difference(dos_array, ref_dos_array,
                             result_array_path=Path(working_dir) / "diff_dos.npy")
