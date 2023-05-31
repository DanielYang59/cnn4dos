#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot COHP and occlusion experiment result (dxy/dxz orbitals).
"""


fermi_level = -2.06264337  # DEBUG
import warnings
warnings.warn("Fermi level set manually.")

from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["DejaVu Sans"]


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


    # Create subplots
    rcParams["axes.linewidth"] = 2.5
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    ax1.plot(xs[0], ys[0], color="black", linewidth=2)
    ax2.plot(xs[1], ys[1], color="black", linewidth=2)
    ax3.plot(xs[2], ys[2], color="black", linewidth=2)
    ax4.plot(xs[3], ys[3], color="black", linewidth=2)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim([-11, 6])
        ax.tick_params(axis="both", labelsize=18, width=2.5, length=5)
        ax.yaxis.set_major_locator(MaxNLocator(5))  # set number of ticks


    # Set x/y axis labels
    fig.supxlabel("$E-E_\mathrm{f}$ (eV)", fontsize=24)
    ax1.set_ylabel("$\Delta\mathit{E}_{\mathrm{ads}}$ (eV)", fontsize=24)
    ax3.set_ylabel("-COHP", fontsize=24)


    # Adjust spacing of subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.4,
                        bottom=0.15)  # x-axis title position


    # Save figure
    plt.savefig("cohp_vs_occlusion.png", dpi=300)
    plt.show()  # DEBUG
