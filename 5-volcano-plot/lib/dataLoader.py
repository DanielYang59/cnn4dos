

import json
import os
import pandas as pd


class dataLoader:
    def __init__(self) -> None:
        """Data loader class for volcano plots, load:
            1. adsorption energy
            2. reaction pathway
            3. thermal correction (ZPE, TS) for adsorption energy
             
        """
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
    
    
    def load_reaction_pathway(self, file):
        """Load reaction pathway json file.

        Args:
            file (str): path to reaction pathway file

        Returns:
            dict: reaction pathway dict
            
        """
        # Check args
        assert os.path.exists(file) and file.endswith("json")
        
        # Import reaction pathway file
        with open(file) as f:
            reaction_pathway_dict = json.load(f)
            
        return reaction_pathway_dict 
    

# Test area
if __name__ == "__main__":
    path = "../../0-dataset/label_adsorption_energy"
    substrates = ["g-C3N4_is", "nitrogen-graphene_is", "vacant-graphene_is", "C2N_is", "BN_is", "BP_is"]
    adsorbates = ["2-COOH", "3-CO", "4-OCH", "5-OCH2", "6-OCH3", "7-O", "8-OH", "11-H"]
    
    # Test adsorption energy loading
    loader = dataLoader(debug=True)
    loader.load_adsorption_energy(path, substrates, adsorbates)
    # print(loader.adsorption_energy_dict)

