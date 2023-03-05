#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import yaml
import sys
sys.path.insert(0, "../../5-volcano-plot/src/lib")

from dataLoader import dataLoader
from utils import stack_adsorption_energy_dict


def generate_corr_matrix(config_file, corr_type="pearson"):
    """Generate correlation matrix from adsorption energy data.

    Args:
        config_file (str): path to config file
        corr_type (str, optional): method of correlation calculation. Defaults to "pearson".

    Returns:
        pd.DataFrame: correlation coefficient DataFrame
        
    """
    # Load configs
    with open(config_file) as ymlfile:
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
    return adsorption_energy_df.corr(method=corr_type)
