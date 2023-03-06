#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ref: QUT Notebook Page 69. """


import os
import yaml

from src.lib.dataLoader import dataLoader
from src.lib.reactionCalculator import reactionCalculator
from src.lib.scalingRelation import scalingRelation
from src.lib.volcanoPlotter import volcanoPlotter


if __name__ == "__main__":
    # Load configs
    with open("config.yaml") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    adsorption_energy_path = cfg["path"]["adsorption_energy_path"]
    thermal_correction_file = cfg["path"]["thermal_correction_file"]
    adsorbate_energy_file = cfg["path"]["adsorbate_energy_file"] 
    reaction_pathway_file = cfg["path"]["reaction_pathway_file"] 
    
    substrates = cfg["species"]["substrates"]
    adsorbates = cfg["species"]["adsorbates"]
    
    group_x = cfg["reaction"]["group_x"]
    descriptor_x = cfg["reaction"]["descriptor_x"] 
    group_y = cfg["reaction"]["group_y"]
    descriptor_y = cfg["reaction"]["descriptor_y"]
    
    external_potential = cfg["corrections"]["external_potential"]
    
    x_range = cfg["plot"]["x_range"]
    y_range = cfg["plot"]["y_range"]
    markers = cfg["plot"]["markers"]
    label_selection = cfg["plot"]["label_selection"]
    
    
    # Loading adsorption energy
    loader = dataLoader()
    loader.load_adsorption_energy(adsorption_energy_path, substrates, adsorbates)
    
    loader.calculate_adsorption_free_energy(correction_file=thermal_correction_file)
    
    # Calculate adsorption energy linear scaling relations
    calculator = scalingRelation(adsorption_energy_dict=loader.adsorption_free_energy, descriptors=(descriptor_x, descriptor_y), mixing_ratios="AUTO", verbose=False, remove_ads_prefix=True) 

    # Calculate reaction energy scaling relations calculator
    reaction_calculator = reactionCalculator(
        adsorption_energy_scaling_relation=calculator.fitting_paras,
        adsorbate_energy_file=adsorbate_energy_file,
        reaction_pathway_file=reaction_pathway_file,
        external_potential=external_potential,
        )
    
    
    reaction_scaling_relations = {
        "CO2RR_CH4":reaction_calculator.calculate_reaction_scaling_relations(name="CO2RR_CH4"),
        "HER":reaction_calculator.calculate_reaction_scaling_relations(name="HER")
                         }
    
    
    # Initialize volcano plotter
    plotter = volcanoPlotter(reaction_scaling_relations,
                             x_range=x_range,
                             y_range=y_range,
                             descriptors=(descriptor_x, descriptor_y),
                             adsorption_free_energies=loader.adsorption_free_energy,
                             markers=markers,
                             )
    
    # Generate limiting potential volcano plot
    plotter.plot_limiting_potential(reaction_name="CO2RR_CH4", show=True,
                                    label_selection=label_selection,
                                    savename=os.path.join("figures", "limiting_potential_CO2RR_CH4.png"),
                                    )


    # Generate rate determining step plot
    # plotter.plot_rds(reaction_name="CO2RR_CH4")
    
    
    # Generate selectivity volcano plot
    
    