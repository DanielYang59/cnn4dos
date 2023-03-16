#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
from pathlib import Path
import yaml
from src.descriptors import Descriptors
from src.plot_correlation import plot_correlation


if __name__ == "__main__":
    # Load configs
    with open("config.yaml") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        
    adsorption_energy_file = Path(cfg["path"]["adsorption_energy_file"])
    element_descriptor_file = Path(cfg["path"]["element_descriptor_file"])
    electronic_descriptor_file = Path(cfg["path"]["electronic_descriptor_file"])
    
    
    # Load and merge descriptor data
    loader = Descriptors(
        adsorption_energy_file=adsorption_energy_file,
        element_descriptor_file=element_descriptor_file,
        electronic_descriptor_file=electronic_descriptor_file,
    )
    
    merged_descriptors = loader.merged_descriptors
    
    
    # Drop metal/substrate columns for correlation map plotting
    merged_descriptors.drop(labels=["metal", "substrate"], axis=1, inplace=True)
    
    
    # Plot Kendall correlation map
    os.makedirs("figures", exist_ok=True)
    plot_correlation(merged_descriptors, method="kendall", savename=os.path.join("figures", "kendall_corr.png"), show=False, verbose=True)
    