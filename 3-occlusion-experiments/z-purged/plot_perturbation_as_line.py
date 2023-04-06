#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize pre-generated perturbation array.
# DEBUG: confirm y-axis unit.
"""


fermi_source_dir = "../0-dataset/z-supporting-info/fermi_level"

energy_start = -14
energy_end = 6
energy_step = 4000

perturbation_dos_name = "perturbation.npy"
orbital_names = ["s", "$p_y$", "$p_z$", "$p_x$", "$d_{xy}$", "$d_{yz}$", "$d_{z^2}$", "$d_{xz}$", "$d_{x^2-y^2}$"]


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

    
def plot_line(arr, energy_array, savedir="."):
    """Plot perturbation as line.

    Args:
        arr (np.ndarray): perturbation array to be plotted, expect shape in (NEDOS, orbital)
        energy_array (np.ndarray): x coordinate array (energy)
        savedir (str): directory where generated figure will be stored
        
    """
    # Check args
    assert os.path.isdir(savedir)
    assert isinstance(arr, np.ndarray) and isinstance(energy_array, np.ndarray)
    assert arr.shape[1] == 9
    assert arr.shape[0] == energy_array.shape[0]
    assert len(orbital_names) == 9
    
    # Set frame thickness
    mpl.rcParams["axes.linewidth"] = 2
    
    # Generate subplot for each orbital
    fig, axs = plt.subplots(arr.shape[1], sharex=True, figsize=(10, 15))
    
    # Create line plot for each orbital
    for index, orbital_arr in enumerate(arr.transpose()):
        ax = axs[index]
        # Create line plot
        ax.plot(energy_array, orbital_arr, color="black")
        
        # Adjust x range and ticks
        ax.set_xlim(-10, 5)
        ax.set_xticks(np.arange(-10, 5 + 2.5, 2.5))
        ax.tick_params(axis="both", which="major", labelsize=16, width=2.5, length=5)
        
        # Add orbital name to the right
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(orbital_names[index], rotation=0, fontsize=24, loc="center", labelpad=45)
        

    # Use same y range for each orbital (3 ps, 5 ds)
    ## p
    p_min = np.amin(arr[:, 1:4])
    p_max = np.amax(arr[:, 1:4])
    for i in range(1, 4):
        axs[i].set_ylim(p_min *1.2, p_max * 1.2)
    
    ## d
    d_min = np.amin(arr[:, 4:9])
    d_max = np.amax(arr[:, 4:9])
    for i in range(4, 9):
        axs[i].set_ylim(d_min * 1.2, d_max * 1.2)
     
    # Set x/y axis labels
    mpl.rcParams["mathtext.default"] = "regular"  # non Italic as default
    fig.supxlabel("E-E$_f$ (eV)", fontsize=24)
    fig.supylabel("Δ ($eV^2$/states)", fontsize=24)  # DEBUG: confirm unit
    
    # Save figure to file
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "perturbation_line.png"), dpi=150)
    print(f"Perturbation result plotted as line in {os.getcwd()}.")
    

def get_fermi(fermi_source_dir):
    """Get fermi level based on file name from csv file.

    Args:
        fermi_source_file (str): fermi level source dir

    Returns:
        float: fermi level
        
    """
    # Check args
    assert os.path.isdir(fermi_source_dir)
    
    # Compile fermi source file name
    current_dir_pieces = os.getcwd().split(os.sep)
    substrate_name = current_dir_pieces[-3]
    adsorbate_name = current_dir_pieces[-2].split("_")[0]
    state_name = current_dir_pieces[-2].split("_")[1]
    metal_name = current_dir_pieces[-1]
    
    # Import fermi source csv file
    fermi_source_file = os.path.join(fermi_source_dir, f"{substrate_name}-{state_name}.csv")
    assert os.path.exists(fermi_source_file)
    fermi_df = pd.read_csv(fermi_source_file, index_col=0)
    
    # Get fermi level from dataframe
    return float(fermi_df.loc[metal_name][adsorbate_name])


if __name__ == "__main__":
    # Get fermi level
    fermi_source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), fermi_source_dir)
    assert os.path.isdir(fermi_source_dir)
    e_fermi = get_fermi(fermi_source_dir)

    # Check args
    assert os.path.exists(perturbation_dos_name)
    assert isinstance(energy_start, (float, int))
    assert isinstance(energy_end, (float, int))
    assert isinstance(energy_step, int)
    assert isinstance(e_fermi, (int, float))
    
    
    # Import perturbation array
    perturbation_array = np.load(perturbation_dos_name)
    
    # Check perturbation array
    if perturbation_array.shape[1] > 16:
        print(f"Caution! Expected perturbation in shape (NEDOS, orbital), found shape ({perturbation_array.shape[0]}, {perturbation_array.shape[1]})")

    # Generate x coordinates (energy)
    energy_array = np.linspace(energy_start, energy_end, energy_step)
    ## Subtract fermi level
    energy_array -= e_fermi
    
    
    # Plot perturbation array as line
    plot_line(perturbation_array, energy_array)
