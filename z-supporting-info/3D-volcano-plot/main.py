#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from src.RidgeMapPlotter import RidgeMapPlotter

def main():
    plotter = RidgeMapPlotter(
        coordinates=(-156.250305, 18.890695, -154.714966, 20.275080),
        font_url='https://github.com/google/fonts/blob/main/ofl/uncialantiqua/UncialAntiqua-Regular.ttf?raw=true',
        show_plot=False,
        cmap='plasma',  # https://matplotlib.org/stable/gallery/color/colormap_reference.html
        interpolate=True,
        )

    # Generate and plot 3D ridge
    plotter.get_3D_data(num_lines=150)
    plotter.plot_3D_ridge(filename=Path("figures") / "3D_volcano.png")

    # Generate and plot 2D projection
    plotter.get_2D_data(num_lines=150)
    plotter.plot_2D_projection(filename=Path("figures") / "2D_volcano.png", alpha=1.0)

if __name__ == '__main__':
    main()