

import sys
import numpy as np
from scipy import stats
import pandas as pd


def linear_fitting_without_mixing(adsorption_energies, group_x, group_y, descriptor_x, descriptor_y, verbose=True):
    """Perform individual linear fit for two groups of adsorbates.

    Args:
        adsorption_energies (dict): dict of adsorption energies
        group_x (list): group x adsorbates
        group_y (list): group y adsorbates
        descriptor_x (str): x-axis adsorbate
        descriptor_y (str): y-axis adsorbate
        verbose (bool, optional): print results during fitting. Defaults to True.

    Returns:
        tuple: (fit results of group_x, fit results of group_y)
        
    """
    # Check args
    assert isinstance(adsorption_energies, dict)
    assert isinstance(group_x, list)
    assert isinstance(group_y, list)
    assert descriptor_x != descriptor_y
    assert isinstance(descriptor_x, str) and descriptor_x in group_x
    assert isinstance(descriptor_y, str) and descriptor_y in group_y
    assert not list(set(group_x).intersection(group_y))
    
    
    # Add substrate name to row index (prep for stacking)
    for key in adsorption_energies:
        adsorption_energies[key] = adsorption_energies[key].rename(index=lambda x: f'{key}_{str(x)}')
            
    # Stack adsorption energy DataFrames
    stacked_adsorption_energy_df = pd.concat(adsorption_energies.values())

    
    # Perform linear fitting for group x
    fitting_results_x = {}
    for ads in group_x:
        if ads != descriptor_x:
            result = stats.linregress(
                np.array(stacked_adsorption_energy_df[descriptor_x]), 
                np.array(stacked_adsorption_energy_df[ads])
                )  # slope, intercept, r_value, p_value, std_err
            
            fitting_results_x[f"{descriptor_x}_{ads}"] = result
            
            if verbose:
                print(f"R2 for {descriptor_x} vs {ads} is {result.rvalue}.")
    
    
    # Perform linear fitting for group y
    fitting_results_y = {}
    for ads in group_y:
        if ads != descriptor_y:
            result = stats.linregress(
                np.array(stacked_adsorption_energy_df[descriptor_y]), 
                np.array(stacked_adsorption_energy_df[ads])
                )
            
            fitting_results_y[f"{descriptor_y}_{ads}"] = result
            
            if verbose:
                print(f"R2 for {descriptor_y} vs {ads} is {result.rvalue}.")
        
    
    return (fitting_results_x, fitting_results_y)
    

def linear_fitting_with_mixing(adsorption_energies, descriptor_x, descriptor_y, verbose=True):
    # Check args
    assert isinstance(adsorption_energies, dict)
    assert descriptor_x != descriptor_y
    assert isinstance(descriptor_x, str)
    assert isinstance(descriptor_y, str)
    
    
    # Add substrate name to row index (prep for stacking)
    for key in adsorption_energies:
        adsorption_energies[key] = adsorption_energies[key].rename(index=lambda x: f'{key}_{str(x)}')
            
    # Stack adsorption energy DataFrames
    stacked_adsorption_energy_df = pd.concat(adsorption_energies.values())
    
    
    # Loop through adsorbates
    fitting_results = {}
    for ads in list(stacked_adsorption_energy_df.columns.values):
        if ads not in [descriptor_x, descriptor_y]:
            
            # Test mixing percentage
            percentage_dict = {}
            for percentage in range(101):
                # Compile x array
                x_array = np.array(stacked_adsorption_energy_df[descriptor_x]) * percentage * 0.01 + \
                    np.array(stacked_adsorption_energy_df[descriptor_y]) * (100 - percentage) * 0.01
                
                # Perform linear fitting
                result = stats.linregress(
                    x_array, 
                    np.array(stacked_adsorption_energy_df[ads]))
                
                percentage_dict[percentage] = result
                
                if verbose:
                    print(f"Adsorbate: {ads}, {descriptor_x} {percentage} %, {descriptor_y} {100 - percentage} %, r2: {round(result.rvalue, 6)}.")
                    
            
            # Get best and worst percentage
            r2values = [i.rvalue for i in percentage_dict.values()]
            best_r2 = max(r2values)
            best_percentage = r2values.index(best_r2)
            
            print(f"Best R2 for {ads} is {round(best_r2, 4)}, {descriptor_x} {best_percentage} %. Worst R2 for {ads} is {round(min(r2values), 4)}, {descriptor_x} {r2values.index(min(r2values))} %.")
          

            # Perform fitting with best descriptor percentage
            x_array = np.array(stacked_adsorption_energy_df[descriptor_x]) * best_percentage * 0.01 + \
                    np.array(stacked_adsorption_energy_df[descriptor_y]) * (100 - best_percentage) * 0.01
                
            fitting_results[ads] = stats.linregress(
                x_array, 
                np.array(stacked_adsorption_energy_df[ads]))
    
    
    return fitting_results
   
    
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
    
    # linear_fitting_without_mixing(adsorption_energies, group_x, group_y, descriptor_x, descriptor_y)
    linear_fitting_with_mixing(adsorption_energies, descriptor_x, descriptor_y, verbose=False)
    