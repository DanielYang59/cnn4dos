#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from dataLoader import dataLoader
import numpy as np
import os
import pandas as pd
from reactionCalculator import reactionCalculator
from scalingRelation import scalingRelation
from utils import stack_adsorption_energy_dict


class volcanoDebugger:
    def __init__(self, substrates, adsorbates, descriptors, external_potential, adsorption_energy_file, thermal_correction_file, adsorbate_free_energy_file, reaction_pathway_file, debug_dir="debug") -> None:
        # Update attrib
        self.substrates = substrates
        self.adsorbates = adsorbates
        self.descriptors = descriptors
        self.external_potential = external_potential
        
        os.makedirs(debug_dir, exist_ok=True)
        self.debug_dir = debug_dir
        
        
        # Initialize data loader
        loader = dataLoader()
        
        # Load DFT adsorption energy
        loader.load_adsorption_energy(path=adsorption_energy_file, substrates=substrates, adsorbates=adsorbates)
        
        # Calculate adsorption free energy
        loader.calculate_adsorption_free_energy(correction_file=thermal_correction_file)

        self.adsorption_energy = loader.adsorption_energy
        self.adsorption_free_energy = loader.adsorption_free_energy

        # Load free adsorbate energy
        self.adsorbate_free_energy = loader.load_adsorbate_free_energy(path=adsorbate_free_energy_file)
        
        # Load reaction pathways
        self.reaction_pathways = loader.load_reaction_pathway(file=reaction_pathway_file)        
            
                
    def __calculate_adsorption_energy_from_scaling_relation(self, fitting_paras):
        """Calculate adsorption (free) energy from scaling relations.

        Args:
            fitting_paras (dict): scaling relation parameters.

        Returns:
            pd.DataFrame: predicted adsorption energy
            
        """
        # Check arg
        assert isinstance(fitting_paras, dict)
        
        # Load true adsorption energy
        true_adsorption_energy = pd.DataFrame.copy(stack_adsorption_energy_dict(self.adsorption_free_energy, add_prefix=False))
        
        # Use scaling relations to predict adsorption energy (except for descriptors)
        predicted_adsorption_energy = {}
        descriptor_x = np.array(true_adsorption_energy[self.descriptors[0]])
        descriptor_y = np.array(true_adsorption_energy[self.descriptors[1]])
        
        
        for ads in true_adsorption_energy:
            # Get corresponding scaling relation paras
            paras = fitting_paras[ads]
            
            # Make predictions for each adsorbate
            predicted_adsorption_energy[ads] = paras[0] * descriptor_x + paras[1] * descriptor_y + paras[2]
        
        
        predicted_adsorption_energy = pd.DataFrame.from_dict(predicted_adsorption_energy)
                
        return predicted_adsorption_energy
    
    
    def __calculate_adsorption_energy_MAE(self, true_df, predicted_df, exclude_descriptors=True):
        """Calculate MAE of adsorption (free) energy predicted from scaling relations.

        Args:
            true_df (pd.DataFrame): adsorption energy calculated from source
            predicted_df (pd.DataFrame): adsorption energy predicted base on scaling relations
            exclude_descriptors (bool, optional): exclude descriptors from MAE calculation. Defaults to True.
            
        """
        # Check args
        assert isinstance(true_df, pd.DataFrame)
        assert isinstance(predicted_df, pd.DataFrame)
        
        
        # Remove descriptor columns 
        if exclude_descriptors:
            for d in self.descriptors:
                true_df = true_df.drop(d, axis=1)
                predicted_df = predicted_df.drop(d, axis=1)
            
        # Rewrite row names
        predicted_df.index = true_df.index
         
            
        # Calculate adsorption energy difference DataFrame
        diff_df = predicted_df - true_df
        diff_df.to_csv(os.path.join(self.debug_dir, "diff_adsorption_free_energy.csv"))
        
        
        # Calculate mean absolute error (MAE)
        all_results = {}
        for ads, mae in diff_df.abs().mean().items():
            all_results[ads] = mae
            print(f"MAE in adsorption free energy of {ads} is {round(mae, 4)} eV.")
        
        overall_mae = sum(all_results.values()) / len(all_results)
        print(f"Overall MAE is {round(overall_mae, 4)} eV.")
    
    
    def __calculate_limiting_potential(self, method, reaction_name):
        
        ########## Nested Function Starts ##########
        def __calculate_free_energy_for_half_reaction(equation, stacked_adsorption_free_energy):
            # Check args
            assert isinstance(equation, dict)
            assert isinstance(stacked_adsorption_free_energy, pd.DataFrame)
            
            # Initialize free energy constant part
            free_energy_constant_part = 0
            
            # Handle species in half reaction
            for species, num in equation.items():
                # Clean substrates ("*"): skip
                if species == "*":
                    continue
                
                
                # Adsorbed species ("*CO2"): energy = G* + GadsCO2 + GCO2 
                elif species.startswith("*"):
                    species = species.lstrip("*")
                   
                               
                    # Get free species energy
                    free_species_energy = self.adsorbate_free_energy[species]
                    free_energy_constant_part += (num * free_species_energy)
                    
                    # Get adsorption energy dataFrame
                    #print(stacked_adsorption_free_energy)
                    adsorption_energy_df = stacked_adsorption_free_energy[species]
                    
                    #print(adsorption_energy_df)
                    
                
                
                # Proton-electron pairs (PEP)
                elif species == "PEP":
                    pep_energy = 0.5 * self.adsorbate_free_energy["H2"] - self.external_potential
                    free_energy_constant_part += (num * pep_energy)
                    
                    
                # Non-adsorbed species ("CO2"): energy = GCO2
                else:
                    energy = self.adsorbate_free_energy[species.split("_")[0]]  # remove physical state suffix (_l, _g)
                    
                    free_energy_constant_part += (num * energy)
                
            
        
        
        def calculate_limiting_potential_directly(reaction_pathway):
            # Stack adsorption free energy
            print(self.adsorption_free_energy)
            stacked_adsorption_free_energy = stack_adsorption_energy_dict(self.adsorption_free_energy, add_prefix=False)
            
            
            limiting_potential = {}
            
            # Calculate free energy change for each reaction step
            for step_index, equation in reaction_pathway.items():
                # Calculate products energy
                # print(step_index, equation)
                products_energy = __calculate_free_energy_for_half_reaction(equation["products"], stacked_adsorption_free_energy)
                
                
                
                # Calculate reactants energy
                
                
                # Calculate free energy change
                
        
        
        def calculate_limiting_potential_with_scaling_relation(reaction_pathway):
            pass
        
        
        ########## Nested Function Ends ##########
        
        # Check args
        assert method in {"direct", "scaling"}
        
        
        # Load reaction pathway
        assert reaction_name in self.reaction_pathways
        reaction_pathway = self.reaction_pathways[reaction_name]
        reaction_pathway.pop("comment")  # remove comment from reaction pathway dict
        
        
        # Calculate limiting potential directly from energy
        if method == "direct":
            calculate_limiting_potential_directly(reaction_pathway)
        
        
        # Calculate limiting potential directly from scaling relations
        else:
            calculate_limiting_potential_with_scaling_relation(reaction_pathway)
        
        
        return #DEBUG
    
    
    def calculate_adsorption_free_energy_MAE(self, mixing_percentages="AUTO"):
        """Calculate adsorption free energy MAE predicted by scaling relations.

        Args:
            mixing_percentages (str, optional): descriptor mixing ratio. Defaults to "AUTO".
            
        """
        # Calculate adsorption free energy from DFT adsorption energy (true values)
        true_adsorption_free_energy = stack_adsorption_energy_dict(self.adsorption_free_energy, add_prefix=True)
        
        
        # Calculate linear scaling relations of adsorption free energy (predicted values)
        scaling_relation = scalingRelation(
            adsorption_energy_dict=self.adsorption_free_energy,
            mixing_percentages=mixing_percentages,
            descriptors=self.descriptors,
            verbose=False,
        )
        
        predicted_adsorption_free_energy = self.__calculate_adsorption_energy_from_scaling_relation(scaling_relation.fitting_paras)
                
        # Calculate mean absolute error (MAE)
        adsorption_free_energy_MAE = self.__calculate_adsorption_energy_MAE(true_df=true_adsorption_free_energy, predicted_df=predicted_adsorption_free_energy, exclude_descriptors=True)
        
        
    def calculate_limiting_potential_MAE(self, reaction):
        # Calculate limiting potential directly from energy
        self.__calculate_limiting_potential(method="direct", reaction_name=reaction)
        
        
        # Calculate limiting potential using scaling relations
        # self.__calculate_limiting_potential(method="scaling", reaction=reaction) #DEBUG
        
    
    def output_adsorption_free_energy(self):
        """Output adsorption free energies.
        
        Notes:
            1. The adsorption free energies were calculated by applying thermal corrections to adsorbed adsorbates.
        
        """
        # Output adsorption free energy to csf files
        for sub, df in self.adsorption_free_energy.items():
            df.to_csv(os.path.join(self.debug_dir, f"adsorption_free_energy_{sub}.csv"))
    
    
# Test area
if __name__ == "__main__":
    # Initialize debugger
    debugger = volcanoDebugger(
        substrates=["g-C3N4_is", "nitrogen-graphene_is", "vacant-graphene_is", "C2N_is", "BN_is", "BP_is"],
        adsorbates=["2-COOH", "3-CO", "4-OCH", "5-OCH2", "6-OCH3", "7-O", "8-OH", "11-H"],
        descriptors=("3-CO", "8-OH"),
        external_potential=0.17, 
        
        adsorption_energy_file="../../0-dataset/label_adsorption_energy",
        thermal_correction_file="../data/corrections_thermal.csv",
        adsorbate_free_energy_file="../data/energy_adsorbate.csv",
        reaction_pathway_file="../data/reaction_pathway.json",
        
        debug_dir="../debug",
        )
    
    
    # Calculate adsorption free energy
    debugger.output_adsorption_free_energy()
    
    
    # Calculate adsorption free energy MAE (with scaling relations)
    debugger.calculate_adsorption_free_energy_MAE()
    
    
    # Calculate limiting potential MAE
    debugger.calculate_limiting_potential_MAE(reaction="CO2RR_CH4")
    