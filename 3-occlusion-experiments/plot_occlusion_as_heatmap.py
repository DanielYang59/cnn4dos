#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize pre-generated occlusion array.
TODO: confirm y-axis units #DEBUG
"""


energy_start = -14
energy_end = 6
energy_step = 4000
fermi_source_dir = "../0-dataset/z-supporting-info/fermi_level"

occlusion_dos_name = "occlusion.npy"
orbital_names = ["s", "$p_y$", "$p_z$", "$p_x$", "$d_{xy}$", "$d_{yz}$", "$d_{z^2}$", "$d_{xz}$", "$d_{x^2-y^2}$"]
d_orbital_only = True


from matplotlib import rcParams
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import warnings

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["DejaVu Sans"]


def _get_fermi(fermi_source_dir):
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


def plot_heatmap(arr, energy_array, orbital_names, savedir=".", d_orbital_only=False):
    """Plot occlusion array as heatmap.

    Args:
        arr (np.ndarray): occlusion array to be plotted, expect shape in (NEDOS, orbital)
        energy_array (np.ndarray): x coordinate array (energy)
        orbital_names (list): orbital names to be displayed
        savedir (str): directory where generated figure will be stored
        d_orbital_only (bool): plot only d suborbitals

    """
    # Check args
    assert os.path.isdir(savedir)
    assert isinstance(arr, np.ndarray) and isinstance(energy_array, np.ndarray)
    assert arr.shape[1] == 9
    assert arr.shape[0] == energy_array.shape[0]
    assert len(orbital_names) == 9
    assert isinstance(d_orbital_only, bool)


    # Set frame thickness
    mpl.rcParams["axes.linewidth"] = 2


    # Select d orbitals only if required
    if d_orbital_only:
        arr = arr[:, 4:9]
        orbital_names = orbital_names[4:9]

    else:
        warnings.warn("Script intended for d orbital only. Might need to revise subplot arrangement.")


    # Generate subplot for each orbital
    subplot_vspacing = 0.2  # vertical spacing between subplots (in inches)
    fig, axs = plt.subplots(arr.shape[1], sharex=False,
                            figsize=(10, 6 + (5 * subplot_vspacing))
                            )


    # Get min/max values of occlusion array (for a symmetric colorbar)
    vmin = np.amin(arr)
    vmax = np.amax(arr)
    colorbar_range = max(abs(vmin), abs(vmax))
    assert colorbar_range > 0


    # Create line plot for each orbital
    for index, orbital_arr in enumerate(arr.transpose()):
        ax = axs[index]
        # Add 1D heatmap for each orbital
        im = ax.imshow(np.expand_dims(orbital_arr, axis=0),
                           extent=[energy_array[0], energy_array[-1], 0, 2],  # increase 4th var to increase height
                           cmap="coolwarm",  # OR: twilight_shifted
                           vmin=-colorbar_range, vmax=colorbar_range,  # for symmetry colorbar
                           )

        # Adjust x range and ticks
        ax.set_xlim(-10, 5)
        ax.set_xticks(np.arange(-10, 5 + 2.5, 2.5))
        ax.tick_params(axis="both", which="major", labelsize=20, width=2.5, length=5)

        # Add orbital name to the right
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(orbital_names[index], rotation=0, fontsize=24, loc="center", labelpad=40)


    # Adjust vertical spacing between subplots
    plt.subplots_adjust(hspace=0.2,
                        left=0.08,  # move y label closer to the main plot
                        )


    # Hide ticks for x axes
    for i in range(4):
        axs[i].set_xticks([])

    # Hide ticks and labels for y axes
    for i in range(5):
        axs[i].set_yticks([])

    # Set x/y axis labels
    mpl.rcParams["mathtext.default"] = "regular"  # non-Italic as default
    fig.supxlabel("$\mathit{E}-\mathit{E}_f$ (eV)", fontsize=28, x=0.35)
    fig.supylabel("$\Delta\mathit{E}_{ads}$ (eV)", fontsize=28)
    fig.subplots_adjust(bottom=0.12)  # adjust x-axis title position


    # Add colorbar
    cb = fig.colorbar(im, ax=axs.ravel().tolist(),
                      pad=0.15,  # spacing between colorbar and main plot
                      )
    cb.outline.set_visible(False)  # hide border
    cb.ax.tick_params(labelsize=20, width=2.5)  # set tick style
    cb.locator = ticker.MaxNLocator(5)  # set number of ticks
    cb.update_ticks()


    # Save figure
    plt.savefig(os.path.join(savedir, "occlusion_heatmap.png"), dpi=300)
    print(f"Occlusion result plotted as heatmap in {os.getcwd()}.")


if __name__ == "__main__":
    # Get fermi level
    fermi_source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), fermi_source_dir)
    assert os.path.isdir(fermi_source_dir)
    e_fermi = _get_fermi(fermi_source_dir)


    # Check args
    assert os.path.exists(occlusion_dos_name)
    assert isinstance(energy_start, (float, int))
    assert isinstance(energy_end, (float, int))
    assert isinstance(energy_step, int)
    assert isinstance(e_fermi, (int, float))


    # Import occlusion array
    occlusion_array = np.load(occlusion_dos_name)

    # Check occlusion array
    if occlusion_array.shape[1] > 16:
        print(f"Caution! Expected occlusion in shape (NEDOS, orbital), found shape ({occlusion_array.shape[0]}, {occlusion_array.shape[1]})")

    # Generate x coordinates (energy)
    energy_array = np.linspace(energy_start, energy_end, energy_step)
    ## Subtract fermi level
    energy_array -= e_fermi


    # Plot occlusion array as heatmap
    plot_heatmap(occlusion_array, energy_array, orbital_names, d_orbital_only=d_orbital_only)
