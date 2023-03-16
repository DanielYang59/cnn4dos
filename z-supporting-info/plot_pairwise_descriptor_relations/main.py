#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import yaml
from src.descriptors import Descriptors
from src.pairplot import PairPlot


if __name__ == "__main__":
    # Load configs
    with open("config.yaml") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        
    adsorption_energy_file = Path(cfg["path"]["adsorption_energy_file"])
    element_descriptor_file = Path(cfg["path"]["element_descriptor_file"])
    electronic_descriptor_file = Path(cfg["path"]["electronic_descriptor_file"])
    descriptors = cfg["plot"]["descriptors"]
    
    
    # Load and merge descriptor data
    loader = Descriptors(
        adsorption_energy_file=adsorption_energy_file,
        element_descriptor_file=element_descriptor_file,
        electronic_descriptor_file=electronic_descriptor_file
        )
    
    merged_descriptors = loader.merged_descriptors
    
    
    # Generate pairwise relations plot
    plotter = PairPlot(data=merged_descriptors)
    plotter.plot_pairplot(descriptors, savename=Path("figures/pairplot.png"), show=True)
    