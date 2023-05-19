#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["DejaVu Sans"]
import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == "__main__":
    # Get orbital index from user input
    orbital_index = int(input("Which orbital to plot (index starts from \"ZERO\")?"))
    assert 0 <= orbital_index <= 15


    # Read var from generating script
    from generate_shifting import shift_value, elements
    shift_energy_array = np.arange(-shift_value, shift_value + 0.005, 0.005)
    result_dict = {}
    for e in elements:
        result_dict[e] = np.load(os.path.join(f"orbital_{orbital_index}", f"{e}.npy"))


    # Get max and min for all subplots (to share the same colorbar)
    # vmax = max(np.amax(arr) for arr in result_dict.values())
    # vmin = min(np.amin(arr) for arr in result_dict.values())
    # print(f"Min shifting is {vmin} eV, max shifting is {vmax} eV.")

    if vmax > 0 and vmin < 0:
        vmax = max(abs(vmax), abs(vmin))  # symmetry colorbar
        vmin = -vmax
    else:
        raise ValueError("Please manually check value range.")


    # Generate subplot for each element
    fig, axs = plt.subplots(len(result_dict), sharex=True,
                            # figsize=(15, len(result_dict) * 5),
                            )

    for index, ax in enumerate(axs.flat):
        # Fetch data
        key = list(result_dict.keys())[index]
        y = np.expand_dims(np.array(result_dict[key]), axis=0)

        # Create 1D heatmap
        # ref: https://stackoverflow.com/questions/45841786/creating-a-1d-heat-map-from-a-line-graph
        im = ax.imshow(y, extent=[shift_energy_array[0], shift_energy_array[-1], 0, 1.5],
                       cmap="viridis",
                       vmin=vmin, vmax=vmax,
                      )

        ax.set_aspect(0.1)  # x/y ratio (decrease to make subplot wider)

        # Set x tick thickness
        ax.xaxis.set_tick_params(width=2)

        # Y axis settings
        ax.set_yticks([])  # Hide y labels
        ax.set_ylabel(key.split("-")[-1], rotation=0, fontsize=16, loc="bottom", labelpad=30) # Set y title

        # Hide top/left/right frames
        ax.spines.top.set_visible(False)
        ax.spines.left.set_visible(False)
        ax.spines.right.set_visible(False)

    # Add colorbar
    # Ref: https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
    cb = fig.colorbar(im, ax=axs.ravel().tolist())
    cb.set_label('$\Delta\mathit{E}_{\mathrm{ads}}\ \mathrm{(eV)}$', fontsize=16)
    cb.ax.tick_params(labelsize=10, width=2)  # set ticks
    # cb.set_ticks([-20, -10, 0, 10, 20])
    cb.outline.set_visible(False)  # hide border

    # Set x ticks and title
    plt.xticks([-1, -0.5, 0, 0.5, 1], fontsize=12)
    plt.xlabel("DOS Shift (eV)", fontsize=16)


    # plt.tight_layout()
    plt.savefig(f"shift_experiment_orbital_{orbital_index}.png", dpi=300)
    plt.show()
