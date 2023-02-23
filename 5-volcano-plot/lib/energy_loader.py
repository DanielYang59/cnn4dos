

import os
import pandas as pd


class energyLoader:
    def __init__(self, debug=False, debug_dir="debug") -> None:
        """Adsorption energy loading and preprocessing.

        Args:
            debug (bool, optional): debug mode, save all intermediate data to "debug" dir. Defaults to False.
        """
        # Update attrib
        self.debug = debug
        self.debug_dir = debug_dir
    
    
    def load_adsorption_energy(self, path, substrates, adsorbates):
        """Load adsorption energy from csv file (without ZPE corrections).

        Args:
            path (path): path of adsorption energy storage directory.
            substrates (list): list of substrates to be loaded
            adsorbates (list): list of adsorbates to be loaded

        Returns:
            dict: dict of adsorption energies in pd.DataFrame, key is substrate name
            
        """
        # Check args
        assert os.path.isdir(path)
        assert isinstance(substrates, list)
        assert isinstance(adsorbates, list)

        # Load adsorption energy by substrate
        self.adsorption_energy_dict = {
            sub: pd.read_csv(os.path.join(path, f"{sub}.csv"), index_col=0).loc[:, adsorbates]  # apply adsorbate filter
            for sub in substrates
        }
    
    
    def add_thermal_correction(self, correction_file):
        """Add thermal corrections to adsorption energies.

        Args:
            correction_file (str): path to thermal correction csv file
            
        """
        # Check args
        assert os.path.exists(correction_file)
        
        # Import thermal correction file
        df = pd.read_csv(correction_file)
        thermal_correction_dict = dict(zip(df["Species"], df["Correction"].astype(float)))
        

        # Apply thermal correction for each substrate
        free_energy_dict = {}
        for sub, df in self.adsorption_energy_dict.items():
            free_energy_df = pd.DataFrame.copy(df)
            
            for ads in free_energy_df.columns.values:
                # Check if all adsorbates have a correction entry
                if f'*{ads.split("-")[-1]}' not in thermal_correction_dict:
                    raise ValueError(f"Cannot find thermal correction for {ads.split('-')[-1]}")
                
                # Apply correction by column
                free_energy_df[ads] += thermal_correction_dict[f'*{ads.split("-")[-1]}']
            
            free_energy_dict[sub] = free_energy_df
        
        # Update attrib
        self.free_energy_dict = free_energy_dict
        
        
        # debug mode: output free energy to file
        if self.debug:
            print(f"debug mode on. free energy would be output to {self.debug_dir}")
            os.makedirs(self.debug_dir, exist_ok=True)
            for substrate_name, df in free_energy_dict.items():
                df.to_csv(os.path.join(self.debug_dir, f"free_energy_{substrate_name}.csv"))
            

def stack_diff_sub_energy_dict(energy_dict, add_prefix=True):
    """Stack adsorption energies from different substrates vertically.

    Args:
        add_prefix (bool, optional): add substrate to row name as prefix. Defaults to True.
        
    """
    # Check args
    assert isinstance(energy_dict, dict)
    
    # Add substrate name to row index (prep for stacking)
    if add_prefix:
        for key in energy_dict:
            energy_dict[key] = energy_dict[key].rename(index=lambda x: f'{key}_{str(x)}')
            
    # Stack adsorption energy DataFrames
    return pd.concat(energy_dict.values())
        

# Test area
if __name__ == "__main__":
    path = "../../0-dataset/label_adsorption_energy"
    substrates = ["g-C3N4_is", "nitrogen-graphene_is", "vacant-graphene_is", "C2N_is", "BN_is", "BP_is"]
    adsorbates = ["2-COOH", "3-CO", "4-OCH", "5-OCH2", "6-OCH3", "7-O", "8-OH", "11-H"]
    
    # Test adsorption energy loading
    loader = energyLoader(debug=True)
    loader.load_adsorption_energy(path, substrates, adsorbates)
    # print(loader.adsorption_energy_dict)
    
    # Test add ZPE corrections
    loader.add_thermal_correction(correction_file="../data/corrections_thermal.csv")
