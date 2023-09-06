#!/bin/usr/python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

class ShiftPlotter:

    def __init__(self, all_predictions, config):
        self.all_predictions = all_predictions  # Dictionary of all predictions
        self.shift_range = config['shifting']['shifting_range']  # range of shifting, e.g., [-0.5, 0.5]
        self.shift_step = config['shifting']['shifting_step']  # e.g., 0.005
        self.shifting_orbitals = config['shifting']['shifting_orbitals']

    def plot(self):
        # Generate shift_energy_array dynamically
        shift_energy_array = np.arange(self.shift_range[0], self.shift_range[1] + self.shift_step, self.shift_step)

        # Generate subplot for each folder's prediction
        fig, axs = plt.subplots(len(self.all_predictions), sharex=True)

        for index, (folder_name, prediction) in enumerate(self.all_predictions.items()):
            ax = axs[index]
            y = np.expand_dims(prediction, axis=0)  # The prediction array for this folder

            # Create 1D heatmap
            im = ax.imshow(y, extent=[shift_energy_array[0], shift_energy_array[-1], 0, 1.5],
                        cmap="magma",
                        aspect='auto'
                        )

            # Set x tick thickness
            ax.xaxis.set_tick_params(width=2)

            # Y axis settings
            ax.set_yticks([])  # Hide y labels
            ax.set_ylabel(folder_name, rotation=0, fontsize=16, loc="bottom", labelpad=30) # Set y title

            # Hide top/left/right frames
            ax.spines.top.set_visible(False)
            ax.spines.left.set_visible(False)
            ax.spines.right.set_visible(False)

        # Add colorbar
        cb = fig.colorbar(im, ax=axs.ravel().tolist())
        cb.set_label("$\Delta\mathit{E}_{\mathrm{ads}}\ \mathrm{(eV)}$", fontsize=16)
        cb.ax.tick_params(labelsize=10, width=2)  # set ticks
        cb.outline.set_visible(False)  # hide border

        # Set x ticks and title
        plt.xticks(np.arange(self.shift_range[0], self.shift_range[1] + self.shift_step, step=0.1), fontsize=12)
        plt.xlabel("eDOS Shift (eV)", fontsize=16)

        plt.savefig("shift_experiment_plot.png", dpi=300)
        plt.show()
