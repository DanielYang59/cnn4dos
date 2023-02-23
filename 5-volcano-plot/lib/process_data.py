import os
import pandas as pd


def add_thermal_correction(adsorption_energy_dict, correction_file, debug=False, debug_dir="debug"):
    """Add thermal corrections to adsorption energies.

    Args:
        adsorption_energy_dict (dict): adsorption energy dict, key is substrate name, value is adsorption energy pd.DataFrame
        correction_file (str): path to thermal correction csv file
        debug (bool, optional): debug mode, save all intermediate data to "debug" dir. Defaults to False.
        debug_dir (str, optional): debug data saving dir. Defaults to "debug". 
        
    """
    # Check args
    assert isinstance(adsorption_energy_dict, dict)
    assert os.path.exists(correction_file)
    
    # Import thermal correction file
    df = pd.read_csv(correction_file)
    thermal_correction_dict = dict(zip(df["Species"], df["Correction"].astype(float)))
    

    # Apply thermal correction for each substrate
    free_energy_dict = {}
    for sub, df in adsorption_energy_dict.items():
        free_energy_df = pd.DataFrame.copy(df)
        
        for ads in free_energy_df.columns.values:
            # Check if all adsorbates have a correction entry
            if f'*{ads.split("-")[-1]}' not in thermal_correction_dict:
                raise ValueError(f"Cannot find thermal correction for {ads.split('-')[-1]}")
            
            # Apply correction by column
            free_energy_df[ads] += thermal_correction_dict[f'*{ads.split("-")[-1]}']
        
        free_energy_dict[sub] = free_energy_df
    
    
    # debug mode: output free energy to file
    if debug:
        print(f"debug mode on. free energy would be output to {debug_dir}")
        os.makedirs(debug_dir, exist_ok=True)
        for substrate_name, df in free_energy_dict.items():
            df.to_csv(os.path.join(debug_dir, f"free_energy_{substrate_name}.csv"))
            
    # Update attrib
    return free_energy_dict
            