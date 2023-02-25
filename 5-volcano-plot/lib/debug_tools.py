from dataLoader import dataLoader
import os
import pandas as pd
from reactionCalculator import reactionCalculator
from scalingRelation import scalingRelation
from utils import stack_adsorption_energy_dict


class volcanoDebugger:
    def __init__(self) -> None:
        pass
    
    
    def compare_limiting_potential(self):
        pass
    
    
    def calculate_adsorption_free_energy(self, output_dir="debug"):
        # Initialize data loader
        loader = dataLoader()
        
        
        # Load DFT adsorption energy
        loader.load_adsorption_energy(
            path="../../0-dataset/label_adsorption_energy",
            substrates = ["g-C3N4_is", "nitrogen-graphene_is", "vacant-graphene_is", "C2N_is", "BN_is", "BP_is"],
            adsorbates = ["2-COOH", "3-CO", "4-OCH", "5-OCH2", "6-OCH3", "7-O", "8-OH", "11-H"],
        )
                
                
        # Add thermal correction
        loader.calculate_adsorption_free_energy(
            correction_file="../data/corrections_thermal.csv",
        )
        
        
        # Output results
        adsorption_free_energy = loader.adsorption_free_energy
        print(f"Adsorption free energy data saved to directory \"{output_dir}\".")
        os.makedirs(output_dir, exist_ok=True)
        for sub, df in adsorption_free_energy.items():
            df.to_csv(os.path.join(output_dir, f"adsorption_free_energy_{sub}.csv"))
        
    
# Test area
if __name__ == "__main__":
    # Initialize debugger
    debugger = volcanoDebugger()
    
    # Calculate adsorption free energy
    debugger.calculate_adsorption_free_energy()
    