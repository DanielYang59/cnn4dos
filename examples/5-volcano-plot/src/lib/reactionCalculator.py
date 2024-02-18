"""Calculate reaction for volcano plotter."""

import numpy as np

from .dataLoader import dataLoader


class reactionCalculator:
    def __init__(
        self,
        adsorption_energy_scaling_relation,
        adsorbate_energy_file,
        reaction_pathway_file,
        external_potential=0.0,
    ):
        """Calculate scaling relations for reaction.

        Args:
            adsorption_energy_scaling_relation (dict):
                adsorption energy scaling relations dict
            adsorbate_energy_file (str): path to adsorbate energy csv file
            reaction_pathway_file (str): path to reaction pathway json file
            external_potential (float, optional): applied external potential.
                Defaults to 0.0.

        """
        # Update attrib
        assert isinstance(adsorption_energy_scaling_relation, dict)
        self.adsorption_energy_scaling_relation = adsorption_energy_scaling_relation
        assert isinstance(external_potential, (int, float))
        self.external_potential = external_potential

        loader = dataLoader()
        # Load adsorbate energy and reaction pathways
        self.adsorbate_energy = loader.load_adsorbate_free_energy(adsorbate_energy_file)
        # Load reaction pathway
        self.reaction_pathway = loader.load_reaction_pathway(reaction_pathway_file)

    def __calculate_half_reaction(self, half_reaction):
        """Calculate scaling relation parameters for half reactions.

        Args:
            half_reaction (dict): half reaction dict in species:num pairs

        Raises:
            KeyError: if species entry not found

        Returns:
            np.ndarray: parameter array in
                [descriptor_x_para, descriptor_y_para, constant]

        """
        # Check args
        assert isinstance(half_reaction, dict)

        # Initiate empty array
        paras = np.array([0.0, 0.0, 0.0])

        for species, num in half_reaction.items():
            # Skip clean catalysts
            if species == "*":
                continue

            # Proton-electron pairs (PEP)
            elif species == "PEP":
                pep_energy = 0.5 * self.adsorbate_energy["H2"] - self.external_potential
                paras += [0, 0, num * pep_energy]

            # Adsorbed species
            elif species.startswith("*"):
                species = species.lstrip("*")

                # Add free adsorbate energy
                if species in self.adsorbate_energy:
                    paras += [0, 0, num * self.adsorbate_energy[species]]
                else:
                    raise KeyError(f"Cannot find free energy entry for {species}.")

                # Add scaling relation parameters
                paras += num * self.adsorption_energy_scaling_relation[species]

            # non-adsorbate species (free species)
            else:
                # remove physical state suffix (_g, _l)
                species = species.split("_")[0]

                if species in self.adsorbate_energy:
                    paras += [0, 0, num * self.adsorbate_energy[species]]
                else:
                    raise KeyError(f"Cannot find free energy entry for {species}.")

        return paras

    def __calculate_reaction_step(self, equation):
        """Calculate scaling relation parameters for one reaction step.

        Args:
            equation (dict): reaction step dict,
                keys are {"reactions", "products"}

        Returns:
            np.ndarray: parameter array in
                [descriptor_x_para, descriptor_y_para, constant]

        """
        # Check args
        assert isinstance(equation, dict) and len(equation) == 2

        # Calculate half reactions
        products_paras = self.__calculate_half_reaction(equation["products"])
        reactants_paras = self.__calculate_half_reaction(equation["reactants"])

        # Calculate
        return products_paras - reactants_paras

    def calculate_reaction_scaling_relations(self, name):
        """Calculate scaling relations for a selected reaction
            and external potential.

        Args:
            name (str): name of reaction to calculate

        Raises:
            KeyError: if cannot find entry for selected reaction

        Returns:
            dict: scaling relation parameters for each reaction step

        """
        # Check if reaction exists
        if name not in self.reaction_pathway:
            raise KeyError(f"Cannot find data for reaction {name}")

        # Calculate each reaction step
        reaction_paras = {}
        for step_index, equation in self.reaction_pathway[name].items():
            if step_index != "comment":
                reaction_paras[step_index] = self.__calculate_reaction_step(equation)

        return reaction_paras


# Test area
if __name__ == "__main__":
    # Set args
    path = "../../../0-dataset/label_adsorption_energy"
    substrates = [
        "g-C3N4_is",
        "nitrogen-graphene_is",
        "vacant-graphene_is",
        "C2N_is",
        "BN_is",
        "BP_is",
    ]
    adsorbates = ["2-COOH", "3-CO", "4-CHO", "5-CH2O", "6-OCH3", "7-O", "8-OH", "11-H"]

    # Loading adsorption energy
    test_loader = dataLoader()
    test_loader.load_adsorption_energy(path, substrates, adsorbates)

    test_loader.calculate_adsorption_free_energy(
        correction_file="../../data/corrections_thermal.csv"
    )

    # Calculate adsorption energy linear scaling relations
    from .scalingRelation import scalingRelation

    calculator = scalingRelation(
        adsorption_energy_dict=test_loader.adsorption_free_energy,
        descriptors=("3-CO", "8-OH"),
        mixing_ratios="AUTO",
        verbose=False,
        remove_ads_prefix=True,
    )

    # Test reaction energy scaling relations calculator
    reaction_calculator = reactionCalculator(
        adsorption_energy_scaling_relation=calculator.fitting_paras,
        adsorbate_energy_file="../../data/energy_adsorbate.csv",
        reaction_pathway_file="../../data/reaction_pathway.json",
        external_potential=0.17,
    )

    co2rr_para = reaction_calculator.calculate_reaction_scaling_relations(
        name="CO2RR_CH4"
    )
    print(co2rr_para)
