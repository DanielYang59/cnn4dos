#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["DejaVu Sans"]
import matplotlib.pyplot as plt
import numpy as np
import warnings


class dosChangePlotter:
    def __init__(self, project_dir, energy_range, energy_file="energy_for_dos_change.npy", dos_change_file="dos_change.npy"):
        """Plot DOS change from interpolated DOS arrays.

        Args:
            project_dir (Path): directory to scan
            energy_range (tuple): energy (X) range for final DOS change plot.
            energy_file (str, optional): name of energy (X) numpy file. Defaults to "energy_for_dos_change.npy".
            dos_change_file (str, optional): name of DOS change numpy file. Defaults to "dos_change.npy".

        """
        # Check project directory
        assert project_dir.exists()
        self.project_dir = project_dir

        # Update energy and DOS change file names
        self.energy_file = energy_file
        self.dos_change_file = dos_change_file

        # Check energy (X) range
        assert len(energy_range) == 2 and (energy_range[0] < energy_range[1]) and isinstance(energy_range[0], (float, int)) and isinstance(energy_range[1], (float, int))
        self.energy_range = energy_range


        # Get all legal folders
        folders = self.__get_dirs()


        # Loop through all legal folders and generate DOS change plot
        for folder in folders:
            self.__plot_dos_change(folder)


    def __get_dirs(self):
        """Get all legal directories with energy and DOS change files.

        Returns:
            list: list of Path of all legal folders.

        """
        # Get all legal folders
        dirs = []
        for folder in self.project_dir.iterdir():
            if folder.is_dir and not folder.name.startswith("."):
                if (folder / self.energy_file).exists() and (folder / self.dos_change_file).exists():
                    dirs.append(folder)

                else:
                    warnings.warn(f"Can't find core files in {folder}.")

        return dirs


    def __plot_dos_change(self, path):
        """Plot DOS change.

        Args:
            path (Path): working directory.

        """
        # Check path
        assert path.exists()


        # Read energy and DOS change arrays
        energy_array = np.load(path / self.energy_file)
        dos_change_array = np.load(path / self.dos_change_file)


        # Generate for each DOS channel
        fig, axes = plt.subplots(nrows=dos_change_array.shape[1], figsize=(8, 10), sharex=True)

        for index, ax in enumerate(axes):
            ax.plot(energy_array, dos_change_array[:, index], color="black")


        # Set x range
        plt.xlim(self.energy_range)


        # Add x/y titles
        fig.supxlabel(r"$E - E_f$ (eV)", fontsize=20)
        fig.supylabel("$\Delta$DOS", fontsize=20)


        # Save plot
        plt.tight_layout()
        plt.savefig(path / "dos_change.png", dpi=100)
        plt.close()


# Test area
if __name__ == "__main__":
    from pathlib import Path
    plotter = dosChangePlotter(
        project_dir=Path("../results/1-doping"),
        energy_range=(-7.5, 5)
        )
