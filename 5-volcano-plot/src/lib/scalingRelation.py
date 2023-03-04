#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
from .utils import stack_adsorption_energy_dict
import warnings


class scalingRelation:
    def __init__(self, adsorption_energy_dict, descriptors, mixing_ratios, verbose=True, remove_ads_prefix=False):
        """Calculate adsorption energy linear scaling relations.

        Args:
            adsorption_energy_dict (dict): adsorption energy dict, key is substrate, 
                value is pd.DataFrame for adsorption energies
            descriptors (list): [descriptor_x_axis, descriptor_y_axis]
            mixing_ratios (str, tuple): "AUTO" for automatic finding of best ratios,  
                or (x_ratio, y_ratio)
            verbose (bool, optional): verbose. Defaults to True.
            remove_prefix (bool, optional): remove prefix from adsorbate names. Defaults to False.
            
        """
        # Check args
        assert isinstance(adsorption_energy_dict, dict)
        assert len(descriptors) == 2 and (descriptors[0] != descriptors[1])
        assert mixing_ratios == "AUTO" or \
            (isinstance(mixing_ratios, tuple) and len(mixing_ratios) == 2)
        assert isinstance(verbose, bool)
        
        
        # Update attributes
        self.descriptors = descriptors
        self._verbose = verbose
        self._remove_ads_prefix = remove_ads_prefix
        
        # Stack adsorbate energy dataframe of different substrates
        self._stacked_adsorption_energy_df = stack_adsorption_energy_dict(adsorption_energy_dict)
        self.adsorbates = list(self._stacked_adsorption_energy_df.columns.values)      
        
        
        # Automatic mixing ratio fitting
        if mixing_ratios == "AUTO":
            # Test mixing ratio
            mixing_ratio_test_result = {}
            for ratio in range(0, 101):
                mixing_ratio_test_result[ratio] = self.__fit_all_adsorbates_with_given_ratio(
                    ratios=[ratio, 100 - ratio],
                    )

            # Identify best mixing ratio for each adsorbate
            self.best_mixing_ratios = self.__find_best_mixing_ratio(mixing_ratio_test_result)
            
        # Constant mixing ratio fitting
        else:
            self.best_mixing_ratios = {ads:mixing_ratios[0] for ads in self.adsorbates}

        
        # Perform linear fitting with the best ratios
        self.__fit_with_best_ratios()
        
        
        # Translate results into parameters
        self.__fitting_results_to_para()
    
    
    def __find_best_mixing_ratio(self, mixing_ratio_test_result):
        """Find best mixing ratios for each adsorbate.

        Args:
            mixing_ratio_test_result (dict): mixing ratio test result dict

        Returns:
            dict: best mixing ratios for each adsorbate
            
        """
        
        # Check args
        assert isinstance(mixing_ratio_test_result, dict)

        
        # Create list for each adsorbate
        result_dict = {ads: [] for ads in self.adsorbates}
    
        # Loop through ratio
        for p, value in mixing_ratio_test_result.items():
            # Unpack result for each adsorbate
            for ads in self.adsorbates:
                r2 = value[ads].rvalue
                result_dict[ads].append(r2)
        
        
        # Find best ratios for each adsorbate
        best_ratios = {}
        for ads in self.adsorbates:
            # find best mixing ratio
            best = max(result_dict[ads])
            best_index = result_dict[ads].index(best)
            best_ratios[ads] = best_index
            
            
            # verbose
            if self._verbose:
                # find worst mixing ratio
                worst = min(result_dict[ads])
                worst_index = result_dict[ads].index(worst)
                
                # print results
                print(f'Best mixing ratio of "{ads}" is {best_index} % (R2 {round(best, 4)}), worst is {worst_index} % (R2 {round(worst, 4)}).')

            
        return best_ratios
    
    
    def __fit_all_adsorbates_with_given_ratio(self, ratios):
        """Perform linear fitting for ALL adsorbates with: selected two descriptors and given mixing ratios.

        Args:
            ratios (list): [x_descriptor_ratio, y_descriptor_ratio]

        Returns:
            dict: descriptor fitting results, key is adsorbate name, value is linear fitting results
            
        """
        # Check args
        for p in ratios:
            assert 0 <= p <= 100
        assert ratios[0] + ratios[1] == 100
        
        
        # Compile hybrid descriptor
        hybrid_descriptor = (self._stacked_adsorption_energy_df[self.descriptors[0]] * ratios[0] + self._stacked_adsorption_energy_df[self.descriptors[1]] * ratios[1]) * 0.01
        
        # Perform linear fitting for each adsorbate
        return {
                ads: stats.linregress(hybrid_descriptor, self._stacked_adsorption_energy_df[ads])
                for ads in self._stacked_adsorption_energy_df.columns.values
        }
    
    
    def __fit_with_best_ratios(self, ):
        """Do linear fitting for all adsorbates with best ratios found.

        Attrib:
            linear_fitting_results (dict): best fitting results, key is adsorbate name, value is linear fitting results
            
        """
        # Loop through all adsorbates
        results = {}
        for ads in self.adsorbates:            
            # Compile hybrid descriptor array
            ratio = self.best_mixing_ratios[ads]
            assert 0 <= ratio <= 100
            descriptor_x = np.copy(np.array(self._stacked_adsorption_energy_df[self.descriptors[0]]))
            descriptor_y = np.copy(np.array(self._stacked_adsorption_energy_df[self.descriptors[1]]))
            
            hybrid_descriptor_array = (descriptor_x * ratio + descriptor_y * (100 - ratio)) * 0.01
            
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
            a = result.slope * self.best_mixing_ratios[ads] * 0.01
            b = result.slope * (100 - self.best_mixing_ratios[ads]) * 0.01 
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
    calculator = scalingRelation(adsorption_energy_dict=loader.adsorption_free_energy, descriptors=("3-CO", "8-OH"), mixing_ratios="AUTO", verbose=True) 
    print(calculator.fitting_paras) 
    