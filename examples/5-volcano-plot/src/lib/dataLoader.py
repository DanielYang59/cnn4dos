"""Data loader for volcano plotter."""

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

    def calculate_adsorption_free_energy(self, correction_file) -> None:
        """Calculate adsorption free energy from DFT adsorption energy,
            by adding thermal corrections to adsorption energies.

        Args:
            correction_file (str): thermal correction csv file path

         Attrib:
            adsorption_free_energy (dict): dict of adsorption free energies
                in pd.DataFrame, key is substrate name

        Notes:
            thermal correction (ZPE and entropy) of free adsorbates
                and clean substrates would be ignored,
                e.g. GadsCO2 = EadsCO2 + ZPE*CO2 - TS*CO2

        """
        # Check args
        assert os.path.exists(correction_file)

        # Import thermal correction file
        df = pd.read_csv(correction_file)
        thermal_correction_dict = dict(
            zip(df["Species"], df["Correction"].astype(float))
        )

        # Apply thermal correction for each substrate
        self.adsorption_free_energy = {}
        for sub, df in self.adsorption_energy.items():
            free_energy_df = pd.DataFrame.copy(df)

            for ads in free_energy_df.columns.values:
                # Check if all adsorbates have a correction entry
                if f'*{ads.split("-")[-1]}' not in thermal_correction_dict:
                    raise ValueError(
                        f"Cannot find thermal correction for {ads.split('-')[-1]}"
                    )

                # Apply correction by column
                free_energy_df[ads] += thermal_correction_dict[f'*{ads.split("-")[-1]}']

            self.adsorption_free_energy[sub] = free_energy_df

    def load_adsorption_energy(self, path, substrates, adsorbates) -> None:
        """Load adsorption energy from csv file (without ZPE corrections).

        Args:
            path (path): path of adsorption energy storage directory.
            substrates (list): list of substrates to be loaded
            adsorbates (list): list of adsorbates to be loaded

        Attrib:
            adsorption_energy (dict): dict of adsorption energies in
                pd.DataFrame, key is substrate name

        """
        # Check args
        assert os.path.isdir(path)
        assert isinstance(substrates, list)
        assert isinstance(adsorbates, list)

        # Load adsorption energy by substrate
        self.adsorption_energy = {
            sub: pd.read_csv(os.path.join(path, f"{sub}.csv"), index_col=0).loc[
                :, adsorbates
            ]  # apply adsorbate filter
            for sub in substrates
        }

    def load_adsorbate_free_energy(self, path) -> dict:
        """Load adsorbate free energy from csv file.

        Args:
            file (str): path to adsorbate free energy file

        Attrib:
            adsorbate_free_energy (dict): dict of adsorbate-energy pairs

        """
        # Check args
        assert os.path.exists(path) and path.endswith(".csv")

        # Load adsorbate free energy csv file and pack into dict
        df = pd.read_csv(path)

        return dict(zip(df["name"], df["free_energy"].astype(float)))

    def load_reaction_pathway(self, file) -> dict:
        """Load reaction pathway json file.

        Args:
            file (str): path to reaction pathway file

        Attrib:
            reaction_pathway (dict): reaction pathway dict

        """
        # Check args
        assert os.path.exists(file) and file.endswith("json")

        # Import reaction pathway file
        with open(file, encoding="utf-8") as f:
            reaction_pathway = json.load(f)

        return reaction_pathway


# Test area
if __name__ == "__main__":
    test_path = "../../../0-dataset/label_adsorption_energy"
    test_substrates = [
        "g-C3N4_is",
        "nitrogen-graphene_is",
        "vacant-graphene_is",
        "C2N_is",
        "BN_is",
        "BP_is",
    ]
    test_adsorbates = [
        "2-COOH",
        "3-CO",
        "4-CHO",
        "5-CH2O",
        "6-OCH3",
        "7-O",
        "8-OH",
        "11-H",
    ]

    # Test loading adsorption energy
    loader = dataLoader()
    loader.load_adsorption_energy(test_path, test_substrates, test_adsorbates)
    # print(loader.adsorption_energy_dict)

    # Test adding thermal correction
    loader.calculate_adsorption_free_energy(
        correction_file="../../data/corrections_thermal.csv"
    )

    # Test loading reaction pathway
    reaction_pathway = loader.load_reaction_pathway(
        file="../../data/reaction_pathway.json"
    )
    print(reaction_pathway)
