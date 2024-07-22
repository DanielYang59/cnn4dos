"""Plot from saved NumPy array."""

import os
from pathlib import Path
import numpy as np
import sys

# Modify sys.path for shared components
root_dir = Path(__file__).parent
shared_components_dir = root_dir / "../shared_components/src"
sys.path.append(str(shared_components_dir.resolve()))

from src.occlusionPlotter import OcclusionPlotter
from src.utilities import get_fermi_level

from dataLoader import DataLoader


def plot_from_saved():
    # Load config and adsorbate DOS
    data_loader = DataLoader()
    config = data_loader.load_config(root_dir / "config.yaml")

    # Load local predictions
    predictions = np.load(Path(os.getcwd()) / "occlusion_predictions.npy")

    # Read fermi level and plot occlusion
    fermi_level = get_fermi_level(
        working_dir=os.getcwd(),
        fermi_level_source=root_dir / Path(config["path"]["fermi_level_source"]),
    )

    plotter = OcclusionPlotter(predictions, config, fermi_level)
    plotter.plot_heatmap(orbitals=config["plotting"]["heatmap_orbitals"])
    # plotter.plot_line(orbitals=config['plotting']['line_orbitals'])


if __name__ == "__main__":
    plot_from_saved()
