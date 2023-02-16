"""Ref: QUT Notebook Page 69. """


import os
import sys
import yaml
import matplotlib.pyplot as plt

from lib.energy_loader import energyLoader, stack_diff_sub_energy_dict
from lib.fitting import linear_fitting_without_mixing, linear_fitting_with_mixing


if __name__ == "__main__":
    # Load configs
    with open("config.yaml") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    adsorption_energy_path = cfg["path"]["adsorption_energy_path"]
    thermal_correction_file = cfg["path"]["thermal_correction_file"]
    molecule_energy_file = cfg["path"]["molecule_energy_file"]
    
    substrates = cfg["species"]["substrates"]
    adsorbates = cfg["species"]["adsorbates"]
    
    group_x = cfg["reaction"]["group_x"]
    descriptor_x = cfg["reaction"]["descriptor_x"] 
    group_y = cfg["reaction"]["group_y"]
    descriptor_y = cfg["reaction"]["descriptor_y"]
    
    
    # Initialize adsorption energy loader
    energy_loader = energyLoader()
    
    # Load adsorption energies
    energy_loader.load_adsorption_energy(adsorption_energy_path, substrates, adsorbates)

    # Add thermal corrections to adsorption energies
    energy_loader.add_thermal_correction(correction_file=thermal_correction_file)
    
    
    # Perform linear fitting with automatic mixing
    free_energies = stack_diff_sub_energy_dict(energy_loader.free_energy_dict)
    dft_energy_linear_relation = linear_fitting_with_mixing(free_energies, descriptor_x, descriptor_y, verbose=False)
    print(dft_energy_linear_relation)
    
    
    # Generate volcano plot
 
