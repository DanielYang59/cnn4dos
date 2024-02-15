"""Utility scripts for eDOS occlusion experiments."""


from pathlib import Path
from typing import Tuple

import pandas as pd


def get_fermi_level(working_dir: str, fermi_level_source: str) -> float:
    """
    Get Fermi level based on extracted properties from working directory.

    Parameters:
        working_dir (str): The working directory path as a string.
            The expected path format: .../some_folder/{substrate}_{adsorbate}_{state}/{metal}
            Where state should be 'is' for initial state or 'fs' for final state.
            For example: .../some_folder/Si_CO2_is/Al

    Returns:
        float: The Fermi level.
    """
    working_dir = Path(working_dir)
    fermi_level_source = Path(fermi_level_source)

    substrate, adsorbate, state, metal = get_properties_from_path(working_dir)

    csv_file = fermi_level_source / f"{substrate}-{state}.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file {csv_file} not found.")

    df = pd.read_csv(csv_file, index_col=0)

    # Generate a temporary list of adsorbate column names to search for the corresponding adsorbate.
    adsorbate_columns = [i.split("-")[-1] for i in list(df.columns)]

    if adsorbate not in adsorbate_columns:
        raise ValueError(f"Adsorbate {adsorbate} not found in CSV file.")

    original_column_name = df.columns[adsorbate_columns.index(adsorbate)]

    return df.loc[metal, original_column_name]


def get_properties_from_path(working_dir: Path) -> Tuple[str, str, str, str]:
    """
    Extract substrate name, adsorbate name, state ('is' or 'fs'),
    and metal name from a given working directory path.

    Parameters:
        working_dir (Path): The working directory path.
        The expected path format: .../some_folder/{substrate}_{adsorbate}_{state}/{metal}
        Where state should be 'is' for initial state or 'fs' for final state.
        For example: .../some_folder/Si_CO2_is/Al

    Returns:
        tuple: A tuple containing substrate name, adsorbate name, state, and metal name.
    """
    parts = working_dir.parts
    substrate, adsorbate, state = parts[-2].split("_")
    if state not in ["is", "fs"]:
        raise ValueError(
            "State should be 'is' for initial state or 'fs' for final state."
        )
    metal = parts[-1]

    return substrate, adsorbate, state, metal
