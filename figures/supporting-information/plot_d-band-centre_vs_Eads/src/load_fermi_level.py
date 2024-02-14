#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pandas as pd


def load_fermi_level(dosFile, fermi_level_dir):
    """Load fermi level based on DOS file path.

    Args:
        dosFile (str): DOS file path
        fermi_level_dir (str): fermi level csv files storage directory

    Attrib:
        fermi_level (float): fermi level of selected DOS array

    """
    # Check args
    assert os.path.isdir(fermi_level_dir)


    # Unpack info from DOS file name
    ## substrate name
    substrate = dosFile.split(os.sep)[-4]

    ## state name
    state = dosFile.split(os.sep)[-3].split("_")[-1]

    ## Adsorbate name
    adsorbate = dosFile.split(os.sep)[-3].split("_")[0]

    ## metal name
    metal = dosFile.split(os.sep)[-2]


    # Load fermi level csv file
    assert os.path.isdir(fermi_level_dir)
    fermi_level_df = pd.read_csv(os.path.join(fermi_level_dir, f"{substrate}-{state}.csv"), index_col=0)

    # Locate desired fermi level
    return fermi_level_df.loc[metal, adsorbate]
