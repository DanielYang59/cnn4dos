#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for DOS occlusion experiments. Visualize pre-generated gradient array.

"""

fermi_source_dir = "/Users/yang/Library/CloudStorage/OneDrive-QueenslandUniversityofTechnology/0-课题/3-DOS神经网络/1-模型和数据集/2-仅初态预测模型/final-model/dataset/fermi_level"

working_dir = "."
energy_start = -14
energy_end = 6
energy_step = 4000

src_dos_name = "dos_up.npy"
orbital_names = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2-y2"]


import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

    
def plot_line(x_coords, arr, savedir):
    """Traditional line plot.

    Args:
        arr (np.ndarray): DOS array to be plotted, expect shape in (NEDOS, orbital)
        
        savedir (str): directory where generated figure will be stored
        
    """
    # Check args
    assert os.path.isdir(savedir)
    assert isinstance(arr, np.ndarray)
    
    # Generate subplot for each orbital
    fig, axs = plt.subplots(arr.shape[1], sharex=True, figsize=(10, 15))
    
    # Create line plot for each orbital
    for index, orbital_arr in enumerate(arr.transpose()):
        ax = axs[index]
        # Create line plot
        ax.plot(x_coords, orbital_arr)

    # Set x/y axis labels 
    ## Ref: https://matplotlib.org/stable/tutorials/text/mathtext.html
    mpl.rcParams['mathtext.default'] = 'regular'  # do not use Italic as default
    fig.supxlabel("E-E$_f$ (eV)", fontsize=18)
    fig.supylabel("DOS (states/eV)", fontsize=18)
    
    # Save figure to file
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, f"source_DOS.png"), dpi=300)
    # plt.show()


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
    e_fermi = get_fermi(fermi_source_dir)

    # Check args
    assert os.path.isdir(working_dir)
    assert os.path.exists(os.path.join(working_dir, src_dos_name))
    assert isinstance(energy_start, (float, int))
    assert isinstance(energy_end, (float, int))
    assert isinstance(energy_step, int)
    assert isinstance(e_fermi, (int, float))
    
    
    # Import DOS array
    src_dos = np.load(os.path.join(working_dir, src_dos_name))
    
    # Check DOS array
    if src_dos.shape[1] > 16:
        print(f"Caution! Expected DOS in shape (NEDOS, orbital), found shape ({src_dos.shape[0]}, {src_dos.shape[1]})")

    # Generate x coordinates
    energy_array = np.linspace(energy_start, energy_end, energy_step)
    ## Subtract fermi level
    energy_array = energy_array - e_fermi
    
    # Plot original DOS as line  
    plot_line(energy_array, src_dos, savedir=working_dir)
