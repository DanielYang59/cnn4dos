#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
sys.path.insert(0, "../lib")
import copy
import contextlib
from dataLoader import dataLoader
import numpy as np
import os
import pandas as pd
from reactionCalculator import reactionCalculator
from scalingRelation import scalingRelation
import sys
from utils import stack_adsorption_energy_dict
import warnings


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
        
        
        # Calculate scaling relation parameters
        calculator = scalingRelation(adsorption_energy_dict=copy.copy(self.adsorption_free_energy), descriptors=descriptors, mixing_ratios="AUTO", remove_ads_prefix=True, verbose=True)
        self.scaling_relations = calculator.fitting_paras
        
    
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
        true_adsorption_energy = pd.DataFrame.copy(stack_adsorption_energy_dict(self.adsorption_free_energy, add_prefix_to_rowname=False))
        
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
            print(f"MAE in adsorption free energy of {ads} is {mae:.4f} eV.")
        
        overall_mae = sum(all_results.values()) / len(all_results)
        print(f"Overall adsorption energy MAE is {overall_mae:.4f} eV.")
    
    
    def __calculate_limiting_potential(self, method, reaction_name):
        """Calculate limiting potential of selected reaction.

        Args:
            method (str): "direct" for direct calculation from adsorption energy, 
                "scaling" for calculation from scaling relations
            reaction_name (str): name of reaction to calculate

        Returns:
            limiting_potential (pd.Series): limiting potentials
            rate_determining_steps (pd.Series): rate determining steps (starts from 1)
            
        """
        def __calculate_free_energy_for_half_reaction(equation, stacked_adsorption_free_energy):
            """Directly calculate free energy for half of the reaction (either reactants or products).

            Args:
                equation (dict): half reaction equation
                stacked_adsorption_free_energy (pd.DataFrame): stacked adsorption free energy DataFrame

            Notes:
                For half of the reaction, for example *COOH + PEP, the free energy G = G(*COOH) + G(PEP), where the G(*COOH) = G(*) + G(COOH) + Gads(COOH). As a result, the Gads(COOH) term would vary from one sample to the other while other terms is constant. G(*) would cancel out and would be skipped.

            Returns:
                adsorption_energy_df (pd.Series): the diverse part of half reaction energy (each sample has a different energy)
                free_energy_constant_part (float): the constant part of half reaction energy (adsorbate energy is constant)
                
            """
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
                    adsorption_energy_df = stacked_adsorption_free_energy[species]
                    adsorption_energy_df *= num
                    if num != 1:
                        warnings.warn("Adsorbed species num is not one. Please make sure it's intended.")
                    
                
                # Proton-electron pairs (PEP)
                elif species == "PEP":
                    pep_energy = 0.5 * self.adsorbate_free_energy["H2"] - self.external_potential
                    free_energy_constant_part += (num * pep_energy)
                    
                    
                # Non-adsorbed species ("CO2"): energy = GCO2
                else:
                    energy = self.adsorbate_free_energy[species.split("_")[0]]  # remove physical state suffix (_l, _g)
                    
                    free_energy_constant_part += (num * energy)

            
            # When no adsorbed species is present
            if "adsorption_energy_df" not in locals():
                # Generate an all-zero dataframe
                names = copy.copy(stacked_adsorption_free_energy.index.values)
                values = [0 for _ in range(len(names))]
                
                adsorption_energy_df = pd.Series(values, index=names)
                
            return adsorption_energy_df, free_energy_constant_part
            
        
        def calculate_limiting_potential_directly(reaction_pathway):
            """Calculate limiting potential and rate determining step directly from adsorption energy data.

            Args:
                reaction_pathway (dict): reaction pathway dictionary

            Raises:
                ValueError: if sample names of products and reactants are different 

            Returns:
                limiting_potential (pd.Series): limiting potentials
                rate_determining_steps (pd.Series): rate determining steps (starts from 1)
                
            """
            # Stack adsorption free energy (and remove adsorbate name prefix)
            stacked_adsorption_free_energy = stack_adsorption_energy_dict(copy.copy(self.adsorption_free_energy), add_prefix_to_rowname=False, remove_prefix_from_colname=True)
            
            
            # Calculate free energy change for each reaction step
            free_energy_changes = {}
            for step_index, equation in reaction_pathway.items():
                # Calculate reactants energy
                reactant_adsorption_energy, reactant_constant_energy = __calculate_free_energy_for_half_reaction(equation["reactants"], stacked_adsorption_free_energy)
                
                
                # Calculate products energy
                product_adsorption_energy, product_constant_energy = __calculate_free_energy_for_half_reaction(equation["products"], stacked_adsorption_free_energy)
                
                # Reset pd.Series column name
                reactant_adsorption_energy.name = None
                product_adsorption_energy.name = None
                
                
                # Calculate free energy change
                if list(reactant_adsorption_energy.index.values) != list(product_adsorption_energy.index.values):
                    raise ValueError("Inconsistent product and reactant energies found.")
                else:
                    energy_change_df = product_adsorption_energy - reactant_adsorption_energy
                
                
                # Add constant energy (adsorbate energy)
                energy_change_df += (product_constant_energy - reactant_constant_energy)
                
                free_energy_changes[step_index] = energy_change_df

            
            # Pack free energy change dict to DataFrame
            free_energy_changes = pd.DataFrame(free_energy_changes)
            
            # Write free energy changes to file
            free_energy_changes.to_csv(os.path.join(self.debug_dir, "free_energy_change_direct.csv"))
            
            
            # Calculate limiting potential and rate determining steps
            rate_determining_steps = free_energy_changes.idxmax(axis=1)
            limiting_potential = free_energy_changes.max(axis=1)
            
                        
            return limiting_potential, rate_determining_steps
        
        
        def calculate_limiting_potential_from_scaling_relation():
            """Calculate limiting potential and rate determining step based on scaling relations.

            Returns:
                limiting_potential (pd.Series): limiting potentials
                rate_determining_steps (pd.Series): rate determining steps (starts from 1)
                
            """
            # Initialize scaling relation based reaction calculator
            reaction_calculator = reactionCalculator(
                adsorption_energy_scaling_relation=self.scaling_relations,
                adsorbate_energy_file="../../data/energy_adsorbate.csv",  #DEBUG
                reaction_pathway_file="../../data/reaction_pathway.json",  #DEBUG
                external_potential=self.external_potential,
            )
            
            # Calculate free energy scaling relations
            free_energy_scaling_relation = reaction_calculator.calculate_reaction_scaling_relations(name=reaction_name)
                        
            
            # Load two descriptors
            stacked_adsorption_energy = stack_adsorption_energy_dict(self.adsorption_free_energy, add_prefix_to_rowname=False)
            descriptor_x = pd.Series.copy(stacked_adsorption_energy[self.descriptors[0]])
            descriptor_y = pd.Series.copy(stacked_adsorption_energy[self.descriptors[1]])
            
            
            # Calculate free energy change from scaling relations
            free_energy_changes = {}
            for step_index, paras in free_energy_scaling_relation.items():
                free_energy_changes[step_index] = descriptor_x * paras[0] + descriptor_y * paras[1] + paras[2]
            
            
            # Pack free energy change dict to DataFrame
            free_energy_changes = pd.DataFrame(free_energy_changes)
            
            # Write free energy changes to file
            free_energy_changes.to_csv(os.path.join(self.debug_dir, "free_energy_change_scaling.csv"))
            
            
            # Calculate limiting potential and rate determining steps
            rate_determining_steps = free_energy_changes.idxmax(axis=1)
            limiting_potential = free_energy_changes.max(axis=1)
            
            
            return limiting_potential, rate_determining_steps

        
        # Check args
        assert method in {"direct", "scaling"}
        
        
        # Load reaction pathway
        assert reaction_name in self.reaction_pathways
        reaction_pathway = self.reaction_pathways[reaction_name]
        with contextlib.suppress(KeyError):
            reaction_pathway.pop("comment")  # remove comment from reaction pathway dict
        
        
        # Calculate limiting potential directly from energy
        if method == "direct":
             return calculate_limiting_potential_directly(reaction_pathway)
        
        
        # Calculate limiting potential directly from scaling relations
        else:
            return calculate_limiting_potential_from_scaling_relation()    
        
    
    def calculate_adsorption_free_energy_MAE(self, mixing_ratios="AUTO"):
        """Calculate adsorption free energy MAE predicted by scaling relations.

        Args:
            mixing_ratios (str, optional): descriptor mixing ratio. Defaults to "AUTO".
            
        """
        # Calculate adsorption free energy from DFT adsorption energy (true values)
        true_adsorption_free_energy = stack_adsorption_energy_dict(self.adsorption_free_energy, add_prefix_to_rowname=True)
    
      
        # Calculate linear scaling relations of adsorption free energy (predicted values)
        scaling_relation = scalingRelation(
            adsorption_energy_dict=copy.copy(self.adsorption_free_energy),
            mixing_ratios=mixing_ratios,
            descriptors=self.descriptors,
            verbose=False,
        )

        
        predicted_adsorption_free_energy = self.__calculate_adsorption_energy_from_scaling_relation(scaling_relation.fitting_paras)
                
        # Calculate mean absolute error (MAE)
        self.__calculate_adsorption_energy_MAE(true_df=true_adsorption_free_energy, predicted_df=predicted_adsorption_free_energy, exclude_descriptors=True)
                
        
    def calculate_limiting_potential_MAE(self, reaction_name):
        """Calculate limiting potential MAE, comparing direct calculation from adsorption energy and from scaling relations.

        Args:
            reaction_name (str): name of reaction to calculate
            
        """
        # Calculate limiting potential directly from energy
        direct_limiting_potential, direct_rate_determining_steps = self.__calculate_limiting_potential(method="direct", reaction_name=reaction_name)
        
        
        # Calculate limiting potential using scaling relations
        scaling_limiting_potential, scaling_rate_determining_steps = self.__calculate_limiting_potential(method="scaling", reaction_name=reaction_name)
        
        
        # Calculate limiting potential difference
        limiting_potential_diff = direct_limiting_potential - scaling_limiting_potential
       
        
        # Write data to file
        df = pd.concat([direct_limiting_potential, direct_rate_determining_steps, scaling_limiting_potential, scaling_rate_determining_steps, limiting_potential_diff],
                       keys=["direct_limiting_potential", "direct_RDS", "scaling_limiting_potential", "scaling_RDS", "limiting_potential_diff"],
                       axis=1)
        df.to_csv(os.path.join(self.debug_dir, "diff_limiting_potential.csv"))
        
        
        # Calculate limiting potential MAE
        limiting_potential_mae = limiting_potential_diff.abs().mean()
        print(f"MAE of limiting potential calculation is {limiting_potential_mae:.4f} eV.")
        
    
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
        
        adsorption_energy_file="../../../0-dataset/label_adsorption_energy",
        thermal_correction_file="../../data/corrections_thermal.csv",
        adsorbate_free_energy_file="../../data/energy_adsorbate.csv",
        reaction_pathway_file="../../data/reaction_pathway.json",
        
        debug_dir="debug_results",
        )
    
    
    # Calculate adsorption free energy
    debugger.output_adsorption_free_energy()
    
    # Calculate adsorption free energy MAE (with scaling relations)
    debugger.calculate_adsorption_free_energy_MAE()
    
    
    # Calculate limiting potential MAE
    debugger.calculate_limiting_potential_MAE(reaction_name="CO2RR_CH4")
    