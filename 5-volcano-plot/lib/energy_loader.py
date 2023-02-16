

import os
import pandas as pd

class energyLoader:
    def __init__(self) -> None:
        pass
    
    
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
        
        
    def stack_diff_substrates(self, add_prefix=True):
        """Stack adsorption energies from different substrates vertically.

        Args:
            add_prefix (bool, optional): add substrate to row name as prefix. Defaults to True.
            
        """
        # Add substrate name to row index (prep for stacking)
        if add_prefix:
            for key in self.adsorption_energy_dict:
                self.adsorption_energy_dict[key] = self.adsorption_energy_dict[key].rename(index=lambda x: f'{key}_{str(x)}')
                
        # Stack adsorption energy DataFrames
        self.adsorption_energy_dict = pd.concat(self.adsorption_energy_dict.values())
    
    
    def add_corrections(self):
        pass
    
    

# Test area
if __name__ == "__main__":
    path = "../../0-dataset/label_adsorption_energy"
    substrates = ["g-C3N4_is", "nitrogen-graphene_is", "vacant-graphene_is", "C2N_is", "BN_is", "BP_is"]
    adsorbates = ["1-CO2", "2-COOH", "3-CO", "4-OCH", "5-OCH2", "6-OCH3", "7-O", "8-OH", "11-HER"]
    
    loader = energyLoader()
    loader.load_adsorption_energy(path, substrates, adsorbates)
    print(loader.adsorption_energy_dict)
    loader.stack_diff_substrates()
    print(loader.adsorption_energy_dict)
