

import os
import pandas as pd


def load_adsorption_energy(path, substrates, adsorbates):
    """Load adsorption energy from csv file.

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
    return {
        sub: pd.read_csv(os.path.join(path, f"{sub}.csv")).loc[:, adsorbates]  # apply adsorbate filter
        for sub in substrates
    }
    

# Test area
if __name__ == "__main__":
    path = "../../0-dataset/label_adsorption_energy"
    substrates = ["g-C3N4_is", "nitrogen-graphene_is", "vacant-graphene_is", "C2N_is", "BN_is", "BP_is"]
    adsorbates = ["1-CO2", "2-COOH", "3-CO", "4-OCH", "5-OCH2", "6-OCH3", "7-O", "8-OH", "11-HER"]
    
    load_adsorption_energy(path, substrates, adsorbates)
