"""Plot eDOS shifting experiments results."""


from matplotlib import rcParams
import matplotlib.pyplot as plt
from mendeleev import element
import numpy as np
from pathlib import Path

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]


class ShiftPlotter:
    """
    Class for plotting shift experiments.

    Attributes:
        all_predictions (dict): Dictionary containing all prediction data.
        shift_range (list): The range of shift values.
        shift_step (float): The step value for each shift.
        shifting_orbitals (str): The orbital types subject to shifting.
    """

    def __init__(self, all_predictions, config) -> None:
        """
        Initialize ShiftPlotter with prediction data and configuration.

        Parameters:
            all_predictions (dict): Dictionary containing all prediction data.
            config (dict): Dictionary containing various configurations.
        """
        self.all_predictions = all_predictions  # Dictionary of all predictions
        self._sort_predictions_by_element()
        self.shift_range = config["shifting"][
            "shifting_range"
        ]  # range of shifting, e.g., [-0.5, 0.5]
        self.shift_step = config["shifting"]["shifting_step"]  # e.g., 0.005
        self.shifting_orbitals = config["shifting"]["shifting_orbitals"]

    def _sort_predictions_by_element(self) -> None:
        """
        Sorts the 'all_predictions' dictionary based on the atomic numbers of
        the chemical elements found in its keys.
        The atomic numbers are used to order the elements
        according to the periodic Table.
        """
        elements_order = [element(i).symbol for i in range(1, 119)]

        def get_element_from_key(key):
            # Assuming the last component after "-" is the element symbol
            return str(key).split("/")[-1].split("-")[-1]

        sorted_keys = sorted(
            self.all_predictions.keys(),
            key=lambda x: elements_order.index(get_element_from_key(x)),
        )
        self.all_predictions = {k: self.all_predictions[k] for k in sorted_keys}

    def plot(self, colormap="magma", save_dir="figures") -> None:
        """
        Generate a plot based on the provided shift and prediction data.

        Parameters:
            colormap (str, optional): The colormap to use for the plot. Defaults to 'magma'.
            save_dir (str, optional): Directory where to save the figure. Defaults to "figures".

        Returns:
            None: The function saves the plot as a PNG file and displays it.
        """
        # Generate shift_energy_array dynamically
        shift_energy_array = np.arange(
            self.shift_range[0], self.shift_range[1] + self.shift_step, self.shift_step
        )

        # Calculate global vmin and vmax for a symmetric colorbar
        vmax = max(
            abs(np.max(prediction)) for prediction in self.all_predictions.values()
        )
        vmin = -vmax

        # Generate subplot for each folder's prediction
        fig, axs = plt.subplots(len(self.all_predictions), sharex=True)
        if len(self.all_predictions) == 1:
            axs = [axs]  # Make it a list of one axs object

        for index, (folder_name, prediction) in enumerate(self.all_predictions.items()):
            ax = axs[index]
            y = np.expand_dims(
                prediction, axis=0
            )  # The prediction array for this folder

            # Create 1D heatmap
            im = ax.imshow(
                y,
                extent=[shift_energy_array[0], shift_energy_array[-1], 0, 1.5],
                cmap=colormap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )

            # Set x tick thickness
            ax.xaxis.set_tick_params(width=2)

            # Y axis settings
            ax.set_yticks([])  # Hide y labels
            element_name = str(folder_name).split("/")[-1].split("-")[-1]
            ax.set_ylabel(
                element_name, rotation=0, fontsize=16, loc="bottom", labelpad=30
            )  # Set y title

            # Hide top/left/right frames
            ax.spines.top.set_visible(False)
            ax.spines.left.set_visible(False)
            ax.spines.right.set_visible(False)

        # Add colorbar
        cb = fig.colorbar(im, ax=axs)
        cb.set_label("$\Delta\mathit{E}_{\mathrm{ads}}\ \mathrm{(eV)}$", fontsize=16)
        cb.ax.tick_params(labelsize=10, width=2)  # set ticks
        cb.outline.set_visible(False)  # hide border

        # Set x ticks and title
        x_start, x_end = self.shift_range
        x_ticks = np.linspace(x_start, x_end, 5)
        plt.xticks(x_ticks, fontsize=12)
        plt.xlabel("eDOS Shift (eV)", fontsize=16)

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            save_dir
            / f"shift_experiment_{self.shift_range[0]}_{self.shift_range[1]}_{self.shift_step}_{self.shifting_orbitals}.png",
            dpi=600,
        )
