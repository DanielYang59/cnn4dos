#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize shifting array as heatmap, d orbitals only script.
"""

orbital_names = ["$d_{xy}$", "$d_{yz}$", "$d_{z^2}$", "$d_{xz}$", "$d_{x^2-y^2}$"]
optimum_height = {0.1:0.02, 0.5:0.08, 1.0:0.16}


from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["DejaVu Sans"]
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import re


if __name__ == "__main__":
    # Get shifting array filename
    for f in os.listdir("."):
        if f.startswith("shifting_") and f.endswith(".npy"):
            filename = f

    # Get shift value
    try:
        shift_value = float(re.findall("shifting_(\d+\.\d+)eV.npy", filename)[0])
    except IndexError:
        shift_value = float(re.findall("shifting_(\d+)eV.npy", filename)[0])

    # Import shift array in shape (None, orbitals)
    shift_array = np.transpose(np.load(filename))

    # Take d orbitals only
    shift_array = shift_array[:, 4:9]

    ## Get min/max values
    v_min = np.amin(shift_array)
    v_max = np.amax(shift_array)
    if v_min * v_max >=0:  # expect vmin < 0, vmax > 0
        raise ValueError("Please check this min/max manually.")
    ## Rectify min max (for symmetry colorbar)
    vmax = max(abs(v_min), abs(v_max))
    vmin = -vmax


    # Generate x coordinate array
    energy_array = np.arange(-shift_value, shift_value + 0.005, 0.005)


    # Plot differential DOS
    # Generate subplot for each orbital
    fig, axs = plt.subplots(shift_array.shape[1], sharex=True, figsize=(10, 6))

    mpl.rcParams["mathtext.default"] = "regular"


    # Add each orbital
    for index, orbital_arr in enumerate(shift_array.transpose()):
        ax = axs[index]
        # Add 1D heat map
        im = ax.imshow(np.expand_dims(orbital_arr, axis=0),
                           extent=[energy_array[0], energy_array[-1], 0, optimum_height[shift_value]],  # increase last var to increase height
                           # cmap="magma",  # more redish
                           cmap="viridis",  # more greenish
                           vmin=vmin, vmax=vmax,  # use symmetry colorbar
                           )

        # Add orbital name
        ax.set_ylabel(orbital_names[index], rotation=0, fontsize=20, loc="bottom", labelpad=70)

        # Reduce x tick number
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))

        # Hide y ticks
        ax.set_yticks([])

        # Hide ticks for four plots on top
        if index != 4:
            ax.tick_params(axis="both", which="both", length=0)

        # Set x tick size and label size
        else:
            ax.tick_params(axis="both", which="major", labelsize=14)
            ax.xaxis.set_tick_params(width=2)


        # Hide top/left/right frames
        ax.spines.top.set_visible(False)
        ax.spines.left.set_visible(False)
        ax.spines.right.set_visible(False)


    # Add colorbar
    cb = fig.colorbar(im, ax=axs.ravel().tolist())
    cb.set_label("$ΔE_{ads}\ (eV)$", fontsize=24)
    # cb.set_ticks([-20, -10, 0, 10, 20])
    cb.outline.set_visible(False)  # hide border
    cb.ax.tick_params(labelsize=12, width=2.5)  # set ticks

    # Set x/y titles
    fig.supxlabel("Energy Shift (eV)", fontsize=22)

    # Save figure as file
    plt.savefig(f"shifting_{shift_value}eV_heatmap.png", dpi=300, bbox_inches="tight")
    # plt.show()
