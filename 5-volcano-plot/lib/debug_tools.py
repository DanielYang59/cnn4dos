from dataLoader import dataLoader
import os
import pandas as pd
from reactionCalculator import reactionCalculator
from scalingRelation import scalingRelation
from utils import stack_adsorption_energy_dict


class volcanoDebugger:
    def __init__(self, substrates, adsorbates, adsorption_energy_file, thermal_correction_file, adsorbate_free_energy_file, reaction_pathway_file, debug_dir="debug") -> None:
        # Update attrib
        self.substrates = substrates
        self.adsorbates = adsorbates
        
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
        

    def calculate_adsorption_free_energy_MAE(self):
        # Calculate adsorption free energy from DFT adsorption energy
        
        
        # Calculate linear scaling relations of adsorption free energy
        
        # Calculate mean absolute error (MAE)
        
        pass
    
        
        
    def calculate_limiting_potential_MAE(self,):
        pass
    
    
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
        substrates = ["g-C3N4_is", "nitrogen-graphene_is", "vacant-graphene_is", "C2N_is", "BN_is", "BP_is"],
        adsorbates = ["2-COOH", "3-CO", "4-OCH", "5-OCH2", "6-OCH3", "7-O", "8-OH", "11-H"],
        
        adsorption_energy_file="../../0-dataset/label_adsorption_energy",
        thermal_correction_file="../data/corrections_thermal.csv",
        adsorbate_free_energy_file="../data/energy_adsorbate.csv",
        reaction_pathway_file="../data/reaction_pathway.json",
        
        debug_dir="../debug"
        )
    
    
    # Calculate adsorption free energy
    debugger.output_adsorption_free_energy()
    