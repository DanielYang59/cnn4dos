#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os

import pandas as pd


def load_ads_energy(dosFile, ads_energy_dir):
    """Load adsorption energy from DOS file name.

    Args:
        dosFile (str): DOS file name
        ads_energy_dir (str): adsorption energy directory

    Returns:
        float: adsorption energy

    """
    # Check args
    assert os.path.exists(dosFile)
    assert os.path.isdir(ads_energy_dir)

    # Unpack DOS file name
    substrate = dosFile.split(os.sep)[-4]
    adsorbate = dosFile.split(os.sep)[-3].split("_")[0]
    state = dosFile.split(os.sep)[-3].split("_")[-1]
    metal = dosFile.split(os.sep)[-2]

    # Get adsorption energy DataFrame
    df = pd.read_csv(
        os.path.join(ads_energy_dir, f"{substrate}_{state}.csv"), index_col=0
    )

    return df.loc[metal, adsorbate]


# Test area
if __name__ == "__main__":
    load_ads_energy(
        dosFile="../../0-dataset/feature_DOS/BP/3-CO_is/4-Zn/dos_up_65.npy",
        ads_energy_dir="../../0-dataset/label_adsorption_energy",
    )
