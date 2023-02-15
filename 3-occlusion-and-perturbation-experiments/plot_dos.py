#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize extracted DOS array, expect shape (NEDOS, numOrbitals)

"""


e_fermi = 0 

working_dir = "."
energy_start = -14
energy_end = 6
energy_step = 4000

src_dos_name = "dos_up.npy"
orbital_names = ["s", "$p_y$", "$p_z$", "$p_x$", "$d_{xy}$", "$d_{yz}$", "$d_{z^2}$", "$d_{xz}$", "$d_{x^2-y^2}$"]
remove_ghost = True


import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

    
def plot_line(x_coords, arr, savedir):
    """Plot DOS as lines.

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
        ax.plot(x_coords, orbital_arr, color="black")

    # Set x/y axis labels 
    ## Ref: https://matplotlib.org/stable/tutorials/text/mathtext.html
    mpl.rcParams['mathtext.default'] = 'regular'  # do not use Italic as default
    fig.supxlabel("E-E$_f$ (eV)", fontsize=18)
    fig.supylabel("DOS (states/eV)", fontsize=18)
    
    # Save figure to file
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "source_DOS.png"), dpi=50)
    # plt.show()


if __name__ == "__main__":
    # Check args
    assert os.path.isdir(working_dir)
    assert os.path.exists(os.path.join(working_dir, src_dos_name))
    assert isinstance(energy_start, (float, int))
    assert isinstance(energy_end, (float, int))
    assert isinstance(energy_step, int)
    assert isinstance(e_fermi, (int, float))
    
    
    # Import DOS array
    src_dos = np.load(os.path.join(working_dir, src_dos_name))
    
    if remove_ghost:
        print("WARNING! Ghost state will be removed.")
        src_dos[0] = 0.0
    
    # Check DOS array
    if src_dos.shape[1] > 9:
        print(f"Caution! Expected DOS in shape (NEDOS, orbital), found shape ({src_dos.shape[0]}, {src_dos.shape[1]})")


    # Generate x coordinates
    energy_array = np.linspace(energy_start, energy_end, energy_step)
    ## Subtract fermi level
    energy_array = energy_array - e_fermi
    
    
    # Plot original DOS as line  
    plot_line(energy_array, src_dos, savedir=working_dir)
