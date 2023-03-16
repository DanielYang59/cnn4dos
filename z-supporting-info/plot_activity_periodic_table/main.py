#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import yaml
from src.load_limiting_potential import load_limiting_potential
from src.plot_periodic_table import plot_periodic_table


def plot_activity_periodic_table():
     # Load config
    with open("config.yaml") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    limiting_potential_file = cfg["path"]["limiting_potential_file"]
    data_cols = cfg["path"]["data_cols"]
    substrate = cfg["species"]["substrate"]
    
    
    # Load limiting potential for selected substrate
    limiting_potential = load_limiting_potential(limiting_potential_file, substrate=substrate, 
                            name_col=data_cols["name"], potential_col=data_cols["limiting_potential"])
    
    
    # Pass limiting potential to plotter
    plot_periodic_table(
    limiting_potential_dict=limiting_potential,
    extended=False,  # show Lu/Lr elements
    cmap="coolwarm",
    cbar_height=390,  # colorbar height (should reposition to top right)
    alpha=0.80, 
    cbar_location=(0, 162),
    # output_filename=os.path.join("figure", f"pt_{substrate}"),
    show=True, 
    )


if __name__ == "__main__":
   plot_activity_periodic_table()
