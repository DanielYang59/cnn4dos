"""Parse name from eDOS path and load fermi level from csv file."""

import os

import pandas as pd


def load_fermi_level(dosFile, fermi_level_dir):
    """Load fermi level based on eDOS file path.

    Args:
        dosFile (str): eDOS file path
        fermi_level_dir (str): fermi level csv files storage directory

    Attrib:
        fermi_level (float): fermi level of selected eDOS array

    """
    # Check args
    assert os.path.isdir(fermi_level_dir)

    # Unpack info from eDOS file name
    substrate = dosFile.split(os.sep)[-4]  # substrate name
    state = dosFile.split(os.sep)[-3].split("_")[-1]  # state name
    adsorbate = dosFile.split(os.sep)[-3].split("_")[0]  # adsorbate name
    metal = dosFile.split(os.sep)[-2]  # metal name

    # Load fermi level csv file
    assert os.path.isdir(fermi_level_dir)
    fermi_level_df = pd.read_csv(
        os.path.join(fermi_level_dir, f"{substrate}-{state}.csv"), index_col=0
    )

    # Locate desired fermi level
    return fermi_level_df.loc[metal, adsorbate]
