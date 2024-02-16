"""eDOS occlusion experiments plotter: line mode and heatmap mode."""

import os
import warnings
from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]


class OcclusionPlotter:
    """
    Class to generate occlusion heatmaps from predictions.

    Attributes:
        predictions_file (np.ndarray): Path to the file containing occlusion predictions.
        orbital_names (list): List of names for orbitals.
        fermi_level (float): The Fermi energy level.
        dos_energy_range (list): Energy range for density of states (DOS).
        plot_energy_range (list): Energy range for the plot.
    """

    def __init__(self, predictions: np.ndarray, config: dict, fermi_level: float):
        """
        Initialize the OcclusionPlotter with the given parameters.

        Parameters:
            predictions (np.ndarray): the predictions numpy array.
            config (dict): Configuration dictionary for occlusion plot settings.
            fermi_level (float): The Fermi energy level.
        """
        assert isinstance(predictions, np.ndarray)
        self.predictions = predictions
        self.config = config
        self.orbital_names = config["plotting"]["orbital_names"]
        self.fermi_level = fermi_level
        self.dos_energy_range = config["plotting"]["dos_energy_range"]
        self.plot_energy_range = config["plotting"]["plot_energy_range"]

        self._validate_energy_ranges()

    def _validate_energy_ranges(self) -> None:
        """Validate the given energy ranges."""
        if self.dos_energy_range[0] >= self.dos_energy_range[1]:
            raise ValueError(
                "In dos_energy_range, the start energy should be smaller than the end energy."
            )
        if self.plot_energy_range[0] >= self.plot_energy_range[1]:
            raise ValueError(
                "In plot_energy_range, the start energy should be smaller than the end energy."
            )
        if (
            self.plot_energy_range[0] < self.dos_energy_range[0]
            or self.plot_energy_range[1] > self.dos_energy_range[1]
        ):
            raise ValueError("plot_energy_range should be within dos_energy_range.")

    def _take_orbitals(
        self, original_predictions, orbitals, names
    ) -> Tuple[np.ndarray, list]:
        """
        Filter the original predictions based on the selected orbitals.

        Parameters:
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
            "f": [9, 10, 11, 12, 13, 14, 15],
        }

        # Define a dictionary to map orbital types to names
        orbital_names = {
            "s": [names[0]],
            "p": names[1:4],
            "d": names[4:9],
            "f": names[9:16],
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

    def plot_heatmap(
        self,
        orbitals: list = ["s", "p", "d"],
        colormap: str = "viridis",
        show: bool = False,
    ) -> None:
        """
        Plot the heatmap for occlusion predictions.

        Parameters:
            orbitals (list, optional): List of orbitals to consider. Defaults to ["s", "p", "d"].
            colormap (str, optional): The colormap for the heatmap. Defaults to "viridis".
            show (bool, optional): Whether to show the plot or not. Defaults to False.
        """
        # Load predictions
        predictions = self.predictions

        # Take requested orbitals
        array, names = self._take_orbitals(
            predictions, orbitals, names=self.config["plotting"]["orbital_names"]
        )
        if orbitals != ["d"]:
            warnings.warn(
                "This script is intended to work for d orbital only.\
                          Other selection may generate unsatisfactory figures."
            )

        # Create figure and axis
        subplot_vspacing = 0.2  # vertical spacing between subplots
        fig, axs = plt.subplots(
            array.shape[1], sharex=False, figsize=(10, 6 + (5 * subplot_vspacing))
        )

        # Get min/max values of occlusion array (for a symmetric colorbar)
        vmin = np.amin(array)
        vmax = np.amax(array)
        colorbar_range = max(abs(vmin), abs(vmax))
        assert colorbar_range > 0

        # Create line plot for each orbital
        dos_energy_range = self.config["plotting"]["dos_energy_range"]

        for index, orbital_arr in enumerate(array.transpose()):
            ax = axs[index]

            # Adjust for Fermi level
            shifted_dos_range = [
                dos_energy_range[0] - self.fermi_level,
                dos_energy_range[1] - self.fermi_level,
            ]

            # Add 1D heatmap for each orbital
            im = ax.imshow(
                np.expand_dims(orbital_arr, axis=0),
                extent=[
                    shifted_dos_range[0],
                    shifted_dos_range[1],
                    0,
                    2,
                ],  # increase 4th value to increase height
                cmap=colormap,
                vmin=-colorbar_range,
                vmax=colorbar_range,  # symmetric colorbar
            )

            # Adjust x range and ticks
            ax.set_xlim(self.config["plotting"]["plot_energy_range"])
            ax.set_xticks(np.arange(-10, 5 + 2.5, 2.5))
            ax.tick_params(
                axis="both", which="major", labelsize=20, width=2.5, length=5
            )

            # Add orbital names to the right
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(
                names[index], rotation=0, fontsize=36, loc="center", labelpad=48
            )

            # Set border thickness to 2
            for spine in ax.spines.values():
                spine.set_linewidth(2)

        # Adjust vertical spacing between subplots
        plt.subplots_adjust(
            hspace=0.2,
            left=0.08,  # move y label closer to the main plot
        )

        # Hide ticks for x axes
        for i in range(4):
            axs[i].set_xticks([])

        # Hide ticks and labels for y axes
        for i in range(5):
            axs[i].set_yticks([])

        # Set x/y axis labels
        # TODO: increase fontsize further would lead to overlap
        mpl.rcParams["mathtext.default"] = "regular"  # non-Italic
        fig.supxlabel("$\mathit{E}-\mathit{E}_f$ (eV)", fontsize=36, x=0.35)
        fig.supylabel("$\Delta\mathit{E}_{ads}$ (eV)", fontsize=36)
        fig.subplots_adjust(bottom=0.12)  # adjust x-axis title position

        # Add colorbar
        cb = fig.colorbar(
            im,
            ax=axs.ravel().tolist(),
            pad=0.16,  # spacing between colorbar and main plot
        )
        cb.outline.set_visible(False)  # hide border
        cb.ax.tick_params(labelsize=24, width=2.5)  # set tick style
        cb.locator = ticker.MaxNLocator(5)  # set number of ticks
        cb.update_ticks()

        # Save figure and (optionally show figure)
        plt.savefig(Path(os.getcwd()) / "occlusion_heatmap.png", dpi=300)
        if show:
            plt.show()

    def plot_line(self, orbitals: list = ["s", "p", "d"], show: bool = False) -> None:
        """Plot DOS line charts for specified orbitals.

        Args:
            orbitals (list): List of orbitals to plot. Default is ["s", "p", "d"].
            show (bool): Whether to show the plot. Default is False.
        """

        # Filter the predictions based on selected orbitals
        orbital_names = self.config["plotting"]["orbital_names"]
        filtered_predictions, selected_names = self._take_orbitals(
            self.predictions, orbitals, orbital_names
        )

        # Generate subplots
        fig, axs = plt.subplots(len(selected_names), sharex=True, figsize=(10, 15))

        # Plotting Settings
        mpl.rcParams["mathtext.default"] = "regular"  # Non-italic as default
        dos_energy_range = self.config["plotting"]["dos_energy_range"]
        shifted_dos_range = [
            dos_energy_range[0] - self.fermi_level,
            dos_energy_range[1] - self.fermi_level,
        ]
        energy_array = np.linspace(
            shifted_dos_range[0], shifted_dos_range[1], filtered_predictions.shape[0]
        )

        # Plot lines for each selected orbital
        for index, orbital_arr in enumerate(filtered_predictions.transpose()):
            ax = axs[index]
            ax.plot(energy_array, orbital_arr, color="black")
            ax.set_xlim(self.config["plotting"]["plot_energy_range"])
            ax.set_xticks(np.arange(-10, 5 + 2.5, 2.5))
            ax.tick_params(
                axis="both", which="major", labelsize=16, width=2.5, length=5
            )

            # Set border thickness to 2
            for spine in ax.spines.values():
                spine.set_linewidth(2)

            ax.yaxis.set_label_position("right")
            ax.set_ylabel(
                selected_names[index],
                rotation=0,
                fontsize=24,
                loc="center",
                labelpad=45,
            )

        # Add super labels
        fig.supxlabel("$\mathit{E}\ -\ \mathit{E}_f$ (eV)", fontsize=24)
        fig.supylabel("$\Delta\mathit{E}_{ads}$ (eV)", fontsize=24)

        # Save the plot
        plt.savefig(Path.cwd() / "occlusion_line.png", dpi=300)

        if show:
            plt.show()


# Test area
if __name__ == "__main__":
    test_predictions = np.load("../data/g-C3N4_CO_is/4-Co/occlusion_predictions.npy")
    test_config = {
        "plotting": {
            "dos_energy_range": [-14, 6],
            "plot_energy_range": [-10, 5],
            "orbital_names": [
                "s",
                "$p_y$",
                "$p_z$",
                "$p_x$",
                "$d_{xy}$",
                "$d_{yz}$",
                "$d_{z^2}$",
                "$d_{xz}$",
                "$d_{x^2-y^2}$",
            ],
            "heatmap_orbitals": [
                "d",
            ],
            "line_orbitals": ["s", "p", "d"],
        }
    }
    test_fermi_level = -2.06264337

    plotter = OcclusionPlotter(test_predictions, test_config, test_fermi_level)
    plotter.plot_heatmap(
        orbitals=test_config["plotting"]["heatmap_orbitals"], show=True
    )
    plotter.plot_line(orbitals=test_config["plotting"]["line_orbitals"], show=True)
