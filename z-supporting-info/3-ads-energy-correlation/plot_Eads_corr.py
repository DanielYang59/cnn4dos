#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import yaml
import sys
sys.path.insert(0, "../../5-volcano-plot/src/lib")
import matplotlib.pyplot as plt
import seaborn as sns

from dataLoader import dataLoader
from utils import stack_adsorption_energy_dict
from lib.heatmap_revised import corrplot


if __name__ == "__main__":
    # Load configs
    with open("config.yaml") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    adsorption_energy_path = cfg["path"]["adsorption_energy_path"]
    substrates = cfg["species"]["substrates"]
    adsorbates = cfg["species"]["adsorbates"]
    
    
    # Load adsorption energy of selected species
    loader = dataLoader()
    loader.load_adsorption_energy(path=adsorption_energy_path, substrates=substrates, adsorbates=adsorbates)
    
    adsorption_energy_dict = loader.adsorption_energy
    
    
    # Stack adsorption energy from different substrates to one DataFrame
    adsorption_energy_df = stack_adsorption_energy_dict(adsorption_energy_dict, remove_prefix_from_colname=True)
    
    
    # Calculate Pearson correlation coefficient map
    corr_matrix = adsorption_energy_df.corr(method="pearson")
    
    
    # Plot correlation map
    sns.set(color_codes=True, font_scale=1.2)
    plt.figure(figsize=(8, 7))
    corrplot(corr_matrix,
            size_scale=500, marker="s",  # shape of the marker
             )
    
    plt.savefig(os.path.join("figures", "Eads_correlation_map.png"), bbox_inches='tight', dpi=300)
    
    
    # Show correlation coefficients
    print(corr_matrix)    
        