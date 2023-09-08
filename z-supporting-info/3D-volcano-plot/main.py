#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from src.RidgeMapPlotter import RidgeMapPlotter

def main():
    """
    Main function to plot 3D and 2D ridge maps.
    """
    coordinates = (-156.250305, 18.890695, -154.714966, 20.275080)
    font_url = 'https://github.com/google/fonts/blob/main/ofl/uncialantiqua/UncialAntiqua-Regular.ttf?raw=true'

    plotter = RidgeMapPlotter(coordinates, font_url)

    plotter.plot_3D_ridge(Path("figures") / "3D_Volcano.png")
    plotter.plot_2D_projection(Path("figures") / "2D_Volcano.png")

if __name__ == "__main__":
    main()
