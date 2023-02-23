

import os
import pandas as pd
from scipy import stats
from utils import stack_adsorption_energy_dict


class scalingRelation:
    def __init__(self, verbose=True, debug=False, debug_dir="debug") -> None:
        assert isinstance(verbose, bool)
        assert isinstance(debug, bool)
        
        self.verbose = verbose
        self.debug = debug
        self.debug_dir = debug_dir
    
  
    def __calculate_scaling_parameters(self, df, descriptors, mixing_percentages):
        # Check args
        assert isinstance(df, pd.DataFrame)
        assert mixing_percentages == "AUTO" or \
            (isinstance(mixing_percentages, tuple) and len(mixing_percentages) == 2)
        
        
        
        if mixing_percentages == "AUTO":
            # Test mixing percentage
            mixing_percentage_test_result = {}
            for percentage in range(0, 101):
                mixing_percentage_test_result[percentage] = self.__descriptor_fitting(df, 
                                                                descriptors=descriptors,
                                                                percentages=[percentage, 100 - percentage]
                                                                )

            # Identify best mixing percentage for each adsorbate
            best_percentage = self.__find_best_mixing_percentage(mixing_percentage_test_result)

            
            
        # 
        else:
            raise RuntimeError("Not written yet.")
    
    
    def __descriptor_fitting(self, stacked_df, descriptors, percentages):
        """Perform linear fitting with: selected two descriptors and given mixing percentages.

        Args:
            stacked_df (pd.DataFrame): stacked adsorption energy pandas DataFrame
            descriptors (list): [x_axis_descriptor, y_axis_descriptor]
            percentages (list): [x_descriptor_percentage, y_descriptor_percentage]

        Returns:
            dict: descriptor fitting results, key is adsorbate name, value is linear fitting results
            
        """
        # Check args
        assert len(descriptors) == 2 and (descriptors[0] != descriptors[1])
        for p in percentages:
            assert 0 <= p <= 100
        assert percentages[0] + percentages[1] == 100
        
        
        # Compile hybrid descriptor
        hybrid_descriptor = (stacked_df[descriptors[0]] * percentages[0] + stacked_df[descriptors[1]] * percentages[1]) * 0.01
        
        # Perform linear fitting for each adsorbate
        return {
                ads: stats.linregress(hybrid_descriptor, stacked_df[ads])
                for ads in stacked_df.columns.values
        }
    
    
    def __find_best_mixing_percentage(self, mixing_percentage_test_result):
        """Find best mixing percentages for each adsorbate.

        Args:
            mixing_percentage_test_result (dict): mixing percentage test result dict

        Returns:
            dict: best mixing percentages for each adsorbate
            
        """
        
        # Check args
        assert isinstance(mixing_percentage_test_result, dict)

        # Get list of adsorbates
        adsorbates = list(mixing_percentage_test_result[0].keys())
        
        
        # Create list for each adsorbate
        result_dict = {ads: [] for ads in adsorbates}
    
        # Loop through percentage
        for p, value in mixing_percentage_test_result.items():
            # Unpack result for each adsorbate
            for ads in adsorbates:
                r2 = value[ads].rvalue
                result_dict[ads].append(r2)
        
        
        # Find best percentages for each adsorbate
        best_percentages = {}
        for ads in adsorbates:
            # find best mixing percentage
            best = max(result_dict[ads])
            best_index = result_dict[ads].index(best)
            best_percentages[ads] = best_index
            
            
            # verbose
            if self.verbose:
                # find worst mixing percentage
                worst = min(result_dict[ads])
                worst_index = result_dict[ads].index(worst)
                
                # print results
                print(f'Best mixing percentage of "{ads}" is {best_index} % (R2 {round(best, 4)}), worst is {worst_index} % (R2 {round(worst, 4)}).')

            
        return best_percentages


    def calculate_adsorption_energy_scaling_relations(self, adsorption_energy_dict, descriptors,mixing_percentages="AUTO"):
        # Check args
        assert isinstance(adsorption_energy_dict, dict)        
        
        
        # Calculate linear scaling relations from adsorption energies with automatic mixing
        stacked_adsorption_energy_df = stack_adsorption_energy_dict(adsorption_energy_dict)
        
        scaling_relations = self.__calculate_scaling_parameters(
            df=stacked_adsorption_energy_df,
            descriptors=descriptors,
            mixing_percentages=mixing_percentages
            )
        
    
# Test area
if __name__ == "__main__":
    
    path = "../../0-dataset/label_adsorption_energy"
    substrates = ["g-C3N4_is", "nitrogen-graphene_is", "vacant-graphene_is", "C2N_is", "BN_is", "BP_is"]
    adsorbates = ["2-COOH", "3-CO", "4-OCH", "5-OCH2", "6-OCH3", "7-O", "8-OH", "11-H"]
    
    # Loading adsorption energy
    from dataLoader import dataLoader
    loader = dataLoader()
    loader.load_adsorption_energy(path, substrates, adsorbates)
    
    loader.calculate_adsorption_free_energy(correction_file="../data/corrections_thermal.csv")
    
    # 
    calculator = scalingRelation(verbose=True, debug=True)
    
    calculator.calculate_adsorption_energy_scaling_relations(
        adsorption_energy_dict=loader.adsorption_energy,
        descriptors=("3-CO", "8-OH"),
        )
    
    
    