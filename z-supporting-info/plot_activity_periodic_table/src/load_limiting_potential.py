#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pandas as pd


def load_limiting_potential(path, substrate, name_col=0, potential_col=1):
    """Load limiting potential for activity periodic table plot.

    Args:
        path (str): path to limiting potential csv file.
        substrate (str): name of catalyst to plot
        name_col (int, optional): name column index. Defaults to 0.
        potential_col (int, optional): limiting potential column index. Defaults to 1.

    Raises:
        ValueError: if no matched entry found for selected substrate

    Returns:
        dict: {name:limiting_potential} pairs

    """
    # Check args
    assert os.path.exists(path)

    # Load limiting potential csv file
    df = pd.read_csv(path)

    # Get catalyst names and limiting potentials
    names = []
    potentials = []
    for _, row in df.iterrows():
        name = row[name_col]

        if name.split("_")[0] == substrate:  # name comes in "g-C3N4_is_4-Co" format
            names.append(name.split("_")[-1])
            potentials.append(float(row[potential_col]))


    if not names:
        raise ValueError(f"No match found for substrate {substrate}.")
    else:
        return dict(zip(names, potentials))


# Test area
if __name__ == "__main__":
    load_limiting_potential(path="../../../5-volcano-plot/src/test/debug_results/diff_limiting_potential.csv",
                            substrate="g-C3N4")
