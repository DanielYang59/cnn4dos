"""Ref: QUT Notebook Page 69. """


import os
import sys
import yaml
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Load configs
    with open("config.yaml") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    ## paths
    feature_dir = cfg["path"]["feature_dir"]
    
    
    # Load adsorption energies
    adsorption_energies = load_adsorption_energies(path, substrates, adsorbates)
    
    
    # Perform linear fittings by group
    
    
    
    
    # Add corrections
    
    
    
    # Generate volcano plot
 
