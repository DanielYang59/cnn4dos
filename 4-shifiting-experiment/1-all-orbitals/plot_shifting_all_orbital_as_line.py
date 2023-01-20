#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize shifting array as line.

"""


orbital_names = ["s", "$p_y$", "$p_z$", "$p_x$", "$d_{xy}$", "$d_{yz}$", "$d_{z^2}$", "$d_{xz}$", "$d_{x^2-y^2}$"]


import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
    

if __name__ == "__main__":
    # Get shifting array filename
    for f in os.listdir("."):
        if f.startswith("shifting_") and f.endswith(".npy"):
            filename = f
    
    # Get shift value
    shift_value = float(re.findall("shifting_(\d+\.\d+).npy", filename)[0])

    
    # Import shift array
    shift_array = np.transpose(np.load(filename))

    # Generate x coordinates
    energy_array = np.arange(-shift_value, shift_value + 0.005, 0.005)

    
    # Plot shiftings array
    # Set frame thickness
    mpl.rcParams["axes.linewidth"] = 2
    
    # Generate subplot for each orbital
    fig, axs = plt.subplots(9, sharex=True, figsize=(10, 15))
    mpl.rcParams["mathtext.default"] = "regular"
    
    # Add each orbital
    for index, orbital_arr in enumerate(shift_array.transpose()):
        ax = axs[index]
        # Add line
        ax.plot(energy_array, orbital_arr, color="black", linewidth=4)
        
        # Reduce x/y tick number
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        
        # Set tick and label size
        ax.tick_params(axis="both", which="major", labelsize=18, width=2.5, length=5)
        
        # Add orbital name to the right
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(orbital_names[index], rotation=0, fontsize=24, loc="center", labelpad=45)
        
        
    # Use same y range for each orbital (3 ps, 5 ds)
    ## p orbitals
    p_min = np.amin(shift_array[:, 1:4])
    p_max = np.amax(shift_array[:, 1:4])
    for i in range(1, 4):
        axs[i].set_ylim(p_min * 1.2, p_max * 1.2)
    
    ## d orbitals
    d_min = np.amin(shift_array[:, 4:9])
    d_max = np.amax(shift_array[:, 4:9])
    for i in range(4, 9):
        axs[i].set_ylim(d_min * 1.2, d_max * 1.2)
    
    # Set x/y titles
    fig.supxlabel("Energy Shift (eV)", fontsize=24)
    fig.supylabel("$ΔE_{ads}\ (eV)$", fontsize=24)
    
    # Save figure to local file
    plt.tight_layout()
    plt.savefig(f"shifting_{shift_value}eV_line.png", dpi=150)
