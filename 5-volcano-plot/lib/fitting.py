

import numpy as np
from scipy import stats
import pandas as pd


def linear_fitting_without_mixing(adsorption_energies, group_x, group_y, descriptor_x, descriptor_y, verbose=True):
    # Check args
    assert isinstance(adsorption_energies, dict)
    assert isinstance(group_x, list)
    assert isinstance(group_y, list)
    assert isinstance(descriptor_x, str) and descriptor_x in group_x
    assert isinstance(descriptor_y, str) and descriptor_y in group_y
    
    
    # Add substrate name to row index (prep for stacking)
    for key in adsorption_energies:
        adsorption_energies[key] = adsorption_energies[key].rename(index=lambda x: f'{key}_{str(x)}')
            
    # Stack adsorption energy DataFrames
    stacked_adsorption_energy_df = pd.concat(adsorption_energies.values())

    
    # Perform linear fitting for group x
    fitting_results_x = {}
    for ads in group_x:
        if ads != descriptor_x:
            fitting_results_x[f"{descriptor_x}_{ads}"] = stats.linregress(
                np.array(stacked_adsorption_energy_df[descriptor_x]), 
                np.array(stacked_adsorption_energy_df[ads])
                )  # slope, intercept, r_value, p_value, std_err
    
    # Perform linear fitting for group y
    fitting_results_y = {}
    for ads in group_y:
        if ads != descriptor_y:
            fitting_results_y[f"{descriptor_y}_{ads}"] = stats.linregress(
                np.array(stacked_adsorption_energy_df[descriptor_y]), 
                np.array(stacked_adsorption_energy_df[ads])
                )
        
    
    return (fitting_results_x, fitting_results_y)
    
    
# Test area
if __name__ == "__main__":
    # Load adsorption energy
    from data_loader import load_adsorption_energy
    path = "../../0-dataset/label_adsorption_energy"
    substrates = ["g-C3N4_is", "nitrogen-graphene_is", "vacant-graphene_is", "C2N_is", "BN_is", "BP_is"]
    adsorbates = ["1-CO2", "2-COOH", "3-CO", "4-OCH", "5-OCH2", "6-OCH3", "7-O", "8-OH", "11-HER"]
    adsorption_energies = load_adsorption_energy(path, substrates, adsorbates)

    
    # Test fitting
    group_x = ["1-CO2", "2-COOH", "3-CO"]
    descriptor_x = "3-CO"
    group_y = ["4-OCH", "5-OCH2", "6-OCH3", "7-O", "8-OH", "11-HER"]
    descriptor_y = "8-OH"
    
    linear_fitting_without_mixing(adsorption_energies, group_x, group_y, descriptor_x, descriptor_y)
    