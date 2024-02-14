#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
from pathlib import Path
import yaml
from src.load_limiting_potential import load_limiting_potential
from src.plot_periodic_table import plot_periodic_table


def main():
    # Load config
    with open("config.yaml") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    limiting_potential_file = cfg["paths"]["limiting_potential_file"]
    data_cols = cfg["paths"]["data_cols"]
    substrate = cfg["species"]["substrate"]
    colorbar_range = cfg["plottings"]["colorbar_range"]


    # Load limiting potential for selected substrate
    limiting_potential = load_limiting_potential(limiting_potential_file, substrate=substrate,
                            name_col=data_cols["name"], potential_col=data_cols["limiting_potential"])


    # Pass limiting potential to plotter
    os.makedirs("figures", exist_ok=True)
    plot_periodic_table(
    limiting_potential_dict=limiting_potential,
    extended=False,  # hide Lu/Lr elements
    cmap="coolwarm",
    cbar_height=390,  # colorbar height (should reposition to top right)
    alpha=0.80,
    cbar_location=(0, 162),
    cbar_range=colorbar_range,
    output_filename=(Path("figures") / f"{substrate}.png").resolve(),
    show=False
    )


if __name__ == "__main__":
   main()
