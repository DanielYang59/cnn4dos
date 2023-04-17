#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot dxy/dxz orbitals of COHP/occlusion experiment result.

"""


fermi_level = -2.06264337 # DEBUG


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


if  __name__ == "__main__":
    # Import occlusion result
    occ_array = np.load("occlusion.npy")
    occ_energy = np.linspace(-14, 6, 4000)
    occ_energy -= fermi_level
    occ_dxy = occ_array[:, 4]
    occ_dxz = occ_array[:, 7]


    # Import COHP result
    cohp_energy = pd.read_csv("cohp_dxy.dat").iloc[:, 0]
    cohp_dxy = -pd.read_csv("cohp_dxy.dat").iloc[:, 1]
    cohp_dxz = -pd.read_csv("cohp_dxz.dat").iloc[:, 1]

    xs = [occ_energy, cohp_energy, occ_energy, cohp_energy]
    ys = [occ_dxy, cohp_dxy, occ_dxz, cohp_dxz]


    # Generate plot
    mpl.rcParams['axes.linewidth'] = 2.5
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(12, 6))

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(xs[i], ys[i], color="black", linewidth=2)
        plt.xlim([-11, 6])
        plt.xticks(np.arange(-10, 10, 5), fontsize=12)
        plt.yticks(fontsize=12)


    # Set x/y axis labels
    fig.supxlabel(r'E $-$ E$_{\mathrm{f}}$ (eV)', fontsize=20)
    axs[0, 0].set_ylabel(r'$\Delta$E$_{\mathrm{ads}}$', fontsize=20)
    axs[1, 0].set_ylabel('-COHP', fontsize=20)
    for i, ax in enumerate(axs.flatten()):
        ax.tick_params(axis='both', which='major', width=2, length=5)


    # plt.tight_layout()
    plt.savefig("occ_vs_cohp.png", dpi=300)
    plt.show()
