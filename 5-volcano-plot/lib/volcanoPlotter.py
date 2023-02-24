#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class volcanoPlotter:
    def __init__(self, scaling_relations, x_range, y_range, descriptors, adsorption_free_energies):
        # Update attrib
        self.scaling_relations = scaling_relations
        self.x_range = x_range
        self.y_range = y_range
        self.descriptors = descriptors
        self.adsorption_free_energies = adsorption_free_energies
    
    
    def __generate_free_energy_change_mesh(self, reaction_name, density=(400, 400)):
        # Check for requested reaction name
        if reaction_name not in self.scaling_relations:
            raise KeyError(f"Cannot find entry for reaction {reaction_name}")
        
        
        # Generate 2D mesh
        self.xx, self.yy = np.meshgrid(
            np.linspace(self.x_range[0], self.x_range[1], density[0]),
            np.linspace(self.y_range[0], self.y_range[1], density[1]),
            indexing="ij",  
            # outputs are of shape (N, M) for ‘xy’ indexing and (M, N) for ‘ij’ indexing
            )
        
        
        # Get free energy change for each reaction step
        return  {
            step_index: self.xx * paras[0] + self.yy * paras[1] + paras[2]
            # Calculate free energy value for each point on mesh
            for step_index, paras in self.scaling_relations[reaction_name].items()
            }
    
    
    def plot_limititing_potential(self, reaction_name):
        # Generate free energy change mesh for selected reaction
        free_energy_change_mesh = self.__generate_free_energy_change_mesh(reaction_name)
        
        # Stack all meshes of different steps into shape (density_x, density_y, numSteps)
        stacked_mesh = np.stack(list(free_energy_change_mesh.values()), axis=2)
        
        # Get limiting potential mesh (max of free energy change)
        test = np.amax(stacked_mesh, axis=2)
    

        # Create background contour plot
        fig = plt.figure(figsize=[12, 9])
        contour = plt.contourf(self.xx, self.yy, test, 
                               levels=512, cmap="plasma_r", 
                               extend="max", 
                               )
        
        
        # Add colorbar
        cbar = fig.colorbar(contour, shrink=0.95, aspect=15, ticks=[0, 1,2])
        plt.show()
        
        
    
    def plot_rate_determining_step(self):
        pass
    
        
# Test area
if __name__ == "__main__":

    adsorption_energy_path = "../../0-dataset/label_adsorption_energy"
    reaction_pathway_file = "../data/reaction_pathway.json"
    thermal_correction_file = "../data/corrections_thermal.csv"
    adsorbate_energy_file = "../data/energy_adsorbate.csv"
    
    substrates = ["g-C3N4_is", "nitrogen-graphene_is", "vacant-graphene_is", "C2N_is", "BN_is", "BP_is"]
    adsorbates = ["2-COOH", "3-CO", "4-OCH", "5-OCH2", "6-OCH3", "7-O", "8-OH", "11-H"]

    descriptor_x = "3-CO"
    descriptor_y = "8-OH"
    
    external_potential = 0.17
    
    markers = ["o", "^", "s", "d", "P", "*"]
    
    
    # Loading adsorption energy
    from dataLoader import dataLoader
    loader = dataLoader()
    loader.load_adsorption_energy(adsorption_energy_path, substrates, adsorbates)
    
    loader.calculate_adsorption_free_energy(correction_file="../data/corrections_thermal.csv")
    
    # Calculate adsorption energy linear scaling relations
    from scalingRelation import scalingRelation
    calculator = scalingRelation(adsorption_energy_dict=loader.adsorption_free_energy, descriptors=("3-CO", "8-OH"), mixing_percentages="AUTO", verbose=False, remove_ads_prefix=True) 

    # Calculate reaction energy scaling relations calculator
    from reactionCalculator import reactionCalculator
    reaction_calculator = reactionCalculator(
        adsorption_energy_scaling_relation=calculator.fitting_paras,
        adsorbate_energy_file="../data/energy_adsorbate.csv",
        reaction_pathway_file="../data/reaction_pathway.json",
        external_potential=0.17
        )
    
    
    reaction_scaling_relations = {
        "CO2RR_CH4":reaction_calculator.calculate_reaction_scaling_relations(name="CO2RR_CH4"),
        "HER":reaction_calculator.calculate_reaction_scaling_relations(name="HER")
                         }
    
    
    # Generate volcano plot
    plotter = volcanoPlotter(reaction_scaling_relations,
                             x_range=(-5, 0.5),
                             y_range=(-6.5, 0),
                             descriptors=("3-CO", "8-OH"),
                             adsorption_free_energies=loader.adsorption_free_energy
                             # markers=markers, 
                             )
    
    plotter.plot_limititing_potential(reaction_name="CO2RR_CH4")

    
    # plotter.plot_rds(reaction_name="CO2RR_CH4")
    