#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
from .utils import stack_adsorption_energy_dict
import warnings


class scalingRelation:
    def __init__(self, adsorption_energy_dict, descriptors, mixing_percentages, verbose=True, remove_ads_prefix=False):
        """Calculate adsorption energy linear scaling relations.

        Args:
            adsorption_energy_dict (dict): adsorption energy dict, key is substrate, 
                value is pd.DataFrame for adsorption energies
            descriptors (list): [descriptor_x_axis, descriptor_y_axis]
            mixing_percentages (str, tuple): "AUTO" for automatic finding of best percentages,  
                or (x_percentage, y_percentage)
            verbose (bool, optional): verbose. Defaults to True.
            remove_prefix (bool, optional): remove prefix from adsorbate names. Defaults to False.
            
        """
        # Check args
        assert isinstance(adsorption_energy_dict, dict)
        assert len(descriptors) == 2 and (descriptors[0] != descriptors[1])
        assert mixing_percentages == "AUTO" or \
            (isinstance(mixing_percentages, tuple) and len(mixing_percentages) == 2)
        assert isinstance(verbose, bool)
        
        
        # Update attributes
        self.descriptors = descriptors
        self._verbose = verbose
        self._remove_ads_prefix = remove_ads_prefix
        
        # Stack adsorbate energy dataframe of different substrates
        self._stacked_adsorption_energy_df = stack_adsorption_energy_dict(adsorption_energy_dict)
        self.adsorbates = list(self._stacked_adsorption_energy_df.columns.values)      
        
        
        # Automatic mixing percentage fitting
        if mixing_percentages == "AUTO":
            # Test mixing percentage
            mixing_percentage_test_result = {}
            for percentage in range(0, 101):
                mixing_percentage_test_result[percentage] = self.__fit_all_adsorbates_with_given_percentage(
                    percentages=[percentage, 100 - percentage],
                    )

            # Identify best mixing percentage for each adsorbate
            self.best_mixing_percentages = self.__find_best_mixing_percentage(mixing_percentage_test_result)
            
        # Constant mixing percentage fitting
        else:
            self.best_mixing_percentages = {ads:mixing_percentages[0] for ads in self.adsorbates}

        
        # Perform linear fitting with the best percentages
        self.__fit_with_best_percentages()
        
        
        # Translate results into parameters
        self.__fitting_results_to_para()
    
    
    def __find_best_mixing_percentage(self, mixing_percentage_test_result):
        """Find best mixing percentages for each adsorbate.

        Args:
            mixing_percentage_test_result (dict): mixing percentage test result dict

        Returns:
            dict: best mixing percentages for each adsorbate
            
        """
        
        # Check args
        assert isinstance(mixing_percentage_test_result, dict)

        
        # Create list for each adsorbate
        result_dict = {ads: [] for ads in self.adsorbates}
    
        # Loop through percentage
        for p, value in mixing_percentage_test_result.items():
            # Unpack result for each adsorbate
            for ads in self.adsorbates:
                r2 = value[ads].rvalue
                result_dict[ads].append(r2)
        
        
        # Find best percentages for each adsorbate
        best_percentages = {}
        for ads in self.adsorbates:
            # find best mixing percentage
            best = max(result_dict[ads])
            best_index = result_dict[ads].index(best)
            best_percentages[ads] = best_index
            
            
            # verbose
            if self._verbose:
                # find worst mixing percentage
                worst = min(result_dict[ads])
                worst_index = result_dict[ads].index(worst)
                
                # print results
                print(f'Best mixing percentage of "{ads}" is {best_index} % (R2 {round(best, 4)}), worst is {worst_index} % (R2 {round(worst, 4)}).')

            
        return best_percentages
    
    
    def __fit_all_adsorbates_with_given_percentage(self, percentages):
        """Perform linear fitting for ALL adsorbates with: selected two descriptors and given mixing percentages.

        Args:
            percentages (list): [x_descriptor_percentage, y_descriptor_percentage]

        Returns:
            dict: descriptor fitting results, key is adsorbate name, value is linear fitting results
            
        """
        # Check args
        for p in percentages:
            assert 0 <= p <= 100
        assert percentages[0] + percentages[1] == 100
        
        
        # Compile hybrid descriptor
        hybrid_descriptor = (self._stacked_adsorption_energy_df[self.descriptors[0]] * percentages[0] + self._stacked_adsorption_energy_df[self.descriptors[1]] * percentages[1]) * 0.01
        
        # Perform linear fitting for each adsorbate
        return {
                ads: stats.linregress(hybrid_descriptor, self._stacked_adsorption_energy_df[ads])
                for ads in self._stacked_adsorption_energy_df.columns.values
        }
    
    
    def __fit_with_best_percentages(self, ):
        """Do linear fitting for all adsorbates with best percentages found.

        Attrib:
            linear_fitting_results (dict): best fitting results, key is adsorbate name, value is linear fitting results
            
        """
        # Loop through all adsorbates
        results = {}
        for ads in self.adsorbates:            
            # Compile hybrid descriptor array
            percentage = self.best_mixing_percentages[ads]
            assert 0 <= percentage <= 100
            descriptor_x = np.copy(np.array(self._stacked_adsorption_energy_df[self.descriptors[0]]))
            descriptor_y = np.copy(np.array(self._stacked_adsorption_energy_df[self.descriptors[1]]))
            
            hybrid_descriptor_array = (descriptor_x * percentage + descriptor_y * (100 - percentage)) * 0.01
            
            # Perform linear fitting with hybrid descriptor
            results[ads] = stats.linregress(hybrid_descriptor_array, np.array(self._stacked_adsorption_energy_df[ads]))
            
        self.linear_fitting_results = results

    
    def __fitting_results_to_para(self, remove_prefix=False):
        """Translate linear fitting results to parameter arrays.
        
        Attrib:
            fitting_paras (dict): key is adsorbate name, value is [para_descriptor_x, para_descriptor_y, c]
            
        """
        self.fitting_paras = {}
        for ads, result in self.linear_fitting_results.items():
            # Compile three parameters
            a = result.slope * self.best_mixing_percentages[ads] * 0.01
            b = result.slope * (100 - self.best_mixing_percentages[ads]) * 0.01 
            c = result.intercept
            
            # Update para dict
            if self._remove_ads_prefix:
                ads = ads.split("-")[-1]  # remove "X-" from naming of adsorbates "X-CO2"
            self.fitting_paras[ads] = np.array([a, b, c])
            
            # Warn user if R2 score is too low
            if result.rvalue < 0.75:
                warnings.warn(f"R2 fitting of adsorbate {ads} is too low at {result.rvalue}.")

     
# Test area
if __name__ == "__main__":
    # Set args
    path = "../../../0-dataset/label_adsorption_energy"
    substrates = ["g-C3N4_is", "nitrogen-graphene_is", "vacant-graphene_is", "C2N_is", "BN_is", "BP_is"]
    adsorbates = ["2-COOH", "3-CO", "4-OCH", "5-OCH2", "6-OCH3", "7-O", "8-OH", "11-H"]
    
    # Loading adsorption energy
    from .dataLoader import dataLoader
    loader = dataLoader()
    loader.load_adsorption_energy(path, substrates, adsorbates)
    
    loader.calculate_adsorption_free_energy(correction_file="../../data/corrections_thermal.csv")
    
    # Test adsorption energy scaling relations calculator
    calculator = scalingRelation(adsorption_energy_dict=loader.adsorption_free_energy, descriptors=("3-CO", "8-OH"), mixing_percentages="AUTO", verbose=True) 
    print(calculator.fitting_paras) 
    