import pandas as pd


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
