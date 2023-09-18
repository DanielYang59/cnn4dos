#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
import yaml
import warnings

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]

class OcclusionPlotter:
    def __init__(self, predictions_file: Path, config: dict, fermi_level: float):
        """
        Initialize the OcclusionPlotter.

        Parameters:
            predictions_file (Path): The path to the predictions file.
            config (dict): Configuration dictionary for occlusion experiment.
        """
        self.predictions_file = predictions_file
        self.orbital_names = config["plotting"]["orbital_names"]
        self.fermi_level = fermi_level

        self.dos_energy_range = config["plotting"]["dos_energy_range"]
        self.plot_energy_range = config["plotting"]["plot_energy_range"]

        # Check that energy start is smaller than energy end for dos and plot
        if self.dos_energy_range[0] >= self.dos_energy_range[1]:
            raise ValueError("In dos_energy_range, the start energy should be smaller than the end energy.")

        if self.plot_energy_range[0] >= self.plot_energy_range[1]:
            raise ValueError("In plot_energy_range, the start energy should be smaller than the end energy.")

        # Check that plot_energy_range is within dos_energy_range
        if (self.plot_energy_range[0] < self.dos_energy_range[0]) or (self.plot_energy_range[1] > self.dos_energy_range[1]):
            raise ValueError("plot_energy_range should be within dos_energy_range.")

    def _take_orbitals(self, original_predictions, orbitals, names):
        """Filter the original predictions based on the selected orbitals.

        Args:
            original_predictions (np.ndarray): The original prediction data.
            orbitals (list): List containing the orbital types ("s", "p", "d", "f").
            names (list): List containing the orbital names.

        Returns:
            tuple: The filtered prediction data and the corresponding orbital names.
        """
        # Define a dictionary to map orbital types to indices
        orbital_indices = {
            "s": [0],
            "p": [1, 2, 3],
            "d": [4, 5, 6, 7, 8],
            "f": [9, 10, 11, 12, 13, 14, 15]
        }

        # Define a dictionary to map orbital types to names
        orbital_names = {
            "s": [names[0]],
            "p": names[1:4],
            "d": names[4:9],
            "f": names[9:16]
        }

        # Validate that the given orbitals are unique and correct
        assert len(set(orbitals)) == len(orbitals)
        assert all(orb in orbital_indices for orb in orbitals)

        # Collect indices for selected orbitals
        selected_indices = []
        selected_names = []
        for orb in orbitals:
            selected_indices.extend(orbital_indices[orb])
            selected_names.extend(orbital_names[orb])

        # Filter the original predictions based on the selected indices
        filtered_predictions = original_predictions[:, selected_indices]

        return filtered_predictions, selected_names

    def plot_heatmap(self, orbitals: list = ["s", "p", "d"], colormap: str = "viridis", show: bool = False):
        """Plot occlusion predictions as a heatmap."""
        # Load predictions
        predictions = np.load(self.predictions_file)

        # Take requested orbitals
        array, names = self._take_orbitals(predictions, orbitals, names=config["plotting"]["orbital_names"])
        if orbitals != ["d"]:
            warnings.warn("This script is intended to work for d orbital only. Other selection may generate unsatisfactory figures.")

        # Create figure and axis
        subplot_vspacing = 0.2  # vertical spacing between subplots
        fig, axs = plt.subplots(array.shape[1], sharex=False,
                            figsize=(10, 6 + (5 * subplot_vspacing))
                            )

        # Get min/max values of occlusion array (for a symmetric colorbar)
        vmin = np.amin(array)
        vmax = np.amax(array)
        colorbar_range = max(abs(vmin), abs(vmax))
        assert colorbar_range > 0

        # Create line plot for each orbital
        dos_energy_range = config["plotting"]["dos_energy_range"]

        for index, orbital_arr in enumerate(array.transpose()):
            ax = axs[index]

            # Adjust for Fermi level
            shifted_dos_range = [dos_energy_range[0] - self.fermi_level, dos_energy_range[1] - self.fermi_level]

            # Add 1D heatmap for each orbital
            im = ax.imshow(np.expand_dims(orbital_arr, axis=0),
                            extent=[shifted_dos_range[0], shifted_dos_range[1], 0, 2],  # increase 4th value to increase height
                            cmap=colormap,
                            vmin=-colorbar_range, vmax=colorbar_range,  # symmetric colorbar
                            )

            # Adjust x range and ticks
            ax.set_xlim(config["plotting"]["plot_energy_range"])
            ax.set_xticks(np.arange(-10, 5 + 2.5, 2.5))
            ax.tick_params(axis="both", which="major", labelsize=20, width=2.5, length=5)

            # Add orbital names to the right
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(names[index], rotation=0, fontsize=28, loc="center", labelpad=44)

            # Set border thickness to 2
            for spine in ax.spines.values():
                spine.set_linewidth(2)

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
        fig.supxlabel("$\mathit{E}-\mathit{E}_f$ (eV)", fontsize=32, x=0.35)
        fig.supylabel("$\Delta\mathit{E}_{ads}$ (eV)", fontsize=32)
        fig.subplots_adjust(bottom=0.12)  # adjust x-axis title position

        # Add colorbar
        cb = fig.colorbar(im, ax=axs.ravel().tolist(),
                        pad=0.16,  # spacing between colorbar and main plot
                        )
        cb.outline.set_visible(False)  # hide border
        cb.ax.tick_params(labelsize=24, width=2.5)  # set tick style
        cb.locator = ticker.MaxNLocator(5)  # set number of ticks
        cb.update_ticks()

        # Save figure and (optionally show figure)
        plt.savefig(self.predictions_file.parent / "occlusion_heatmap.png", dpi=300)
        if show:
            plt.show()

if __name__ == "__main__":
    # Set the working directory and read config
    working_dir = Path("/Users/yang/Developer/cnn4dos/3-occlusion-experiments/data/g-C3N4_CO_is/4-Co")

    with open("/Users/yang/Developer/cnn4dos/3-occlusion-experiments/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Initialize plotter and plot heatmap
    plotter = OcclusionPlotter(
        predictions_file=working_dir / "occlusion_predictions.npy",
        config=config,
        fermi_level=-2.06264337,
    )
    plotter.plot_heatmap(orbitals=["d", ], show=True)
