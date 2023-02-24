#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataLoader import dataLoader


class reactionCalculator:
    def __init__(self, adsorption_energy_scaling_relation, adsorbate_energy_file, reaction_pathway_file) -> None:
        # Update attrib
        assert isinstance(adsorption_energy_scaling_relation, dict)
        self.adsorption_energy_scaling_relation = adsorption_energy_scaling_relation
        
        
        loader = dataLoader()
        # Load adsorbate energy and reaction pathways
        self.adsorbate_energy = loader.load_adsorbate_free_energy(adsorbate_energy_file)
        # Load reaction pathway
        self.reaction_pathway = loader.load_reaction_pathway(reaction_pathway_file)


    def __calculate_half_reaction(self):
        pass
    
    
    def calculate_reaction(self, name):
        if name not in self.reaction_pathway:
            raise KeyError(f"Cannot find data for reaction {name}")
        
        
        pass
    
    
 
    
    
    
    
    
# Test area
if __name__ == "__main__":
    # Set args
    path = "../../0-dataset/label_adsorption_energy"
    substrates = ["g-C3N4_is", "nitrogen-graphene_is", "vacant-graphene_is", "C2N_is", "BN_is", "BP_is"]
    adsorbates = ["2-COOH", "3-CO", "4-OCH", "5-OCH2", "6-OCH3", "7-O", "8-OH", "11-H"]
    
    # Loading adsorption energy
    from dataLoader import dataLoader
    loader = dataLoader()
    loader.load_adsorption_energy(path, substrates, adsorbates)
    
    loader.calculate_adsorption_free_energy(correction_file="../data/corrections_thermal.csv")
    
    # Calculate adsorption energy linear scaling relations
    from scalingRelation import scalingRelation
    calculator = scalingRelation(adsorption_energy_dict=loader.adsorption_free_energy, descriptors=("3-CO", "8-OH"), mixing_percentages="AUTO", verbose=False) 

    # Test reaction energy scaling relations calculator
    reaction_calculator = reactionCalculator(
        adsorption_energy_scaling_relation=calculator.fitting_paras,
        adsorbate_energy_file="../data/energy_adsorbate.csv",
        reaction_pathway_file="../data/reaction_pathway.json"
        )
    
    reaction_calculator.calculate_reaction(name="CO2RR_CH4")
