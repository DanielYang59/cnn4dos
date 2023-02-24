#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataLoader import dataLoader


class reactionEnergyCalculator:
    def __init__(self, adsorption_energy_scaling_relation) -> None:
        # Check args
        assert isinstance(adsorption_energy_scaling_relation, dict)

    
    
        
    
    
    
    
    
    
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
    reaction_calculator = reactionEnergyCalculator(
        adsorption_energy_scaling_relation=calculator.fitting_paras
    )
