#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import warnings


class dosChangePlotter:
    def __init__(self, project_dir, x_range, energy_file="energy.npy", dos_change_file="dos_change.npy"):
        # Check project directory
        assert project_dir.exists()
        self.project_dir = project_dir

        # Update energy and DOS change file names
        self.energy_file = energy_file
        self.dos_change_file = dos_change_file


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
        # Check path
        assert path.exists()


        # Read energy and DOS change arrays
        energy_array = np.load(path / self.energy_file)
        dos_change_array = np.load(path / self.dos_change_file)


        # Generate for each DOS channel
        fig, axes = plt.subplots(nrows=dos_change_array.shape[1])

        for index, ax in enumerate(axes):
            ax.plot(energy_array, dos_change_array[:, ], color="black")


        plt.show()

        import sys
        sys.exit()


# Test area
if __name__ == "__main__":
    from pathlib import Path
    plotter = dosChangePlotter(
        project_dir=Path("../results/1-doping"),
        x_range=(-5, 5)
        )
