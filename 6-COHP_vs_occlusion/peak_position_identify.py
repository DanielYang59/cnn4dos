#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Identify peak location in COHP/occlusion results.
"""


import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    # Load COHP result
    cohp_energy = pd.read_csv("cohp_dxy.dat").iloc[:, 0]
    cohp_dxz = -pd.read_csv("cohp_dxz.dat").iloc[:, 1]
    cohp_dxy = -pd.read_csv("cohp_dxy.dat").iloc[:, 1]


    # Plot -COHP
    plt.plot(cohp_energy, cohp_dxz, linewidth=2)

    ## Rescale x-axis range
    plt.xlim([-11, 6])

    ## Show plot
    plt.show()
