"""Utils for volcano plotter."""


import pandas as pd


def stack_adsorption_energy_dict(
    energy_dict,
    add_prefix_to_rowname=True,
    remove_prefix_from_colname=False
    ) -> pd.DataFrame:
    """Stack adsorption energies from different substrates vertically.

    Args:
        add_prefix_to_rowname (bool, optional): add substrate to row name as prefix. Defaults to True.
        remove_prefix_from_colname (bool, optional): remove prefix from colname (adsorbate name), for example "2-" in "2-COOH". Defaults to False.

    """
    # Check args
    assert isinstance(energy_dict, dict)

    # Add substrate name to row index (prep for stacking)
    if add_prefix_to_rowname:
        for key in energy_dict:
            energy_dict[key] = energy_dict[key].rename(index=lambda x, key=key: f'{key}_{str(x)}')

    df = pd.concat(energy_dict.values())

    # Remove prefix from column names upon request
    if remove_prefix_from_colname:
        col_names = df.columns.values
        col_names = [i.split("-")[-1] for i in col_names]
        df.columns = col_names

    # Stack adsorption energy DataFrames
    return df


# Test area
if __name__ == "__main__":
    # Set args
    path = "../../../0-dataset/label_adsorption_energy"
    substrates = ["g-C3N4_is", "nitrogen-graphene_is", "vacant-graphene_is", "C2N_is", "BN_is", "BP_is"]
    adsorbates = ["2-COOH", "3-CO", "4-CHO", "5-CH2O", "6-OCH3", "7-O", "8-OH", "11-H"]

    # Loading adsorption energy
    from dataLoader import dataLoader
    loader = dataLoader()
    loader.load_adsorption_energy(path, substrates, adsorbates)

    # Add thermal correction
    loader.calculate_adsorption_free_energy(correction_file="../../data/corrections_thermal.csv")

    # Test stacking adsorption energy dict
    stacked_adsorption_energy = stack_adsorption_energy_dict(
        loader.adsorption_free_energy,
        add_prefix_to_rowname=False,
        remove_prefix_from_colname=True
        )
    print(stacked_adsorption_energy)
