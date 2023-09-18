#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]

class OcclusionPlotter:
    def __init__(self, predictions_file: Path, config: dict, colormap: str = "viridis"):
        """
        Initialize the OcclusionPlotter.

        Parameters:
            predictions_file (Path): The path to the predictions file.
            config (dict): Configuration dictionary for occlusion experiment.
            colormap (str): The colormap to use for the heatmap.
        """
        self.colormap = colormap
        self.predictions_file = predictions_file
        self.orbital_names = config["plotting"]["orbital_names"]

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

    def plot_heatmap(self):
        """Plot occlusion predictions as a heatmap."""
        # Load predictions
        predictions = np.load(self.predictions_file)

        # Create figure and axis
        fig, ax = plt.subplots()

        # Create heatmap
        im = ax.imshow(predictions, cmap=self.colormap)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Predictions", rotation=-90, va="bottom")

        # Set ticks and labels
        ax.set_xticks(np.arange(len(self.orbital_names)))
        ax.set_yticks(np.arange(predictions.shape[0]))
        ax.set_xticklabels(self.orbital_names)
        ax.set_ylabel("Occlusion index")

        # Show the plot
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
        colormap="viridis"
    )
    plotter.plot_heatmap()
