

import csv
import json
import os
import sys #DEBUG


class scalingRelations:
    def __init__(self, free_energy_linear_relation, molecule_energy_file, reaction_pathway_file, external_potential):
        """Calculate and output free energy scaling relations of each reaction step.

        Args:
            free_energy_linear_relation (dict): free energy linear relation dict
            molecule_energy_file (str): path to molecule energy CSV file
            reaction_pathway_file (str): path to reaction pathway JSON file
            external_potential (float, int): external potential in eV

        Notes:
            1. Convert adsorption (free) energy relations to limiting potential relations:
                For reaction step: * + CO2_g + 8 * PEP (proton-electron pair: H+ + e-) -> *COOH + 7 * PEP
                
                The reaction free energy G = [G(*COOH) + 7 * G(PEP)] - [G(*) + G(CO2_g) + 8 * G(PEP)], 
                could reduce to G = [G(*COOH) - G(*)] - [G(CO2_g) + G(PEP)], note the energies of last two molecular species (CO2_g and PEP) is constant.
                
                For the [G(*COOH) - G(*)] part, we have G(*COOH) = G(COOH) + G(*) + GadsCOOH, where GadsCOOH is the adsorption free energy of COOH.
                
                As such, G reduces to G = GadsCOOH + [G(COOH) - G(CO2_g) - G(PEP)].
                
                Meanwhile from the free energy relations, we have: any adsorption free energy is a function of adsorption free energies of two descriptor species, in this case CO and OH, which gives GadsCOOH = a * GadsCO + b * GadsOH + c.
                
                Final the free energy equation reduces to G = [a * GadsCO + b * GadsOH + c] + [G(COOH) - G(CO2_g) - G(PEP)], meaning the free energy of any reaction step is determined by the adsorption free energies of two descriptors. 
            
        """
        # Check args
        assert isinstance(free_energy_linear_relation, dict)
        assert os.path.exists(molecule_energy_file)
        assert os.path.exists(reaction_pathway_file)
        assert isinstance(external_potential, (float, int))
        
        # Update attrib
        self.free_energy_linear_relation = free_energy_linear_relation
        self.external_potential = external_potential
        
        
        # Import molecule energy
        self.molecule_energy_dict = self.import_molecule_energy(molecule_energy_file)
       
        
        # Import reaction path
        self.reaction_pathway_dict = self.import_reaction_pathway(reaction_pathway_file)
        
        
        # Calculate scaling relations for each reaction
        self.scaling_relations_dict = {}
        for reaction_name in self.reaction_pathway_dict:
            self.scaling_relations_dict[reaction_name] = self.calculate_relations(reaction_name)
            sys.exit() #DEBUG
            
            
    def calculate_relations(self, reaction_name):
        """Calculate scaling relation of selected reaction pathway.

        Args:
            reaction_name (str): name of reaction to calculate
            
        """
        
        
        def calculate_energy_term(equation_part, external_potential):
            """Calculate energy term for half of reaction equation.

            Args:
                equation_part (dict): dict of half reaction euqation
                external_potential (float, int): applied external potential in eV
                
            """
            # Check args
            assert isinstance(equation_part, dict)
            assert isinstance(external_potential, (float, int))

            # Calculate energy contribution for each term
            total_energy_contribution = 0
            for species, num in equation_part.items():
                print(species, num) #DEBUG
                # PEP: proton-electron pairs
                if species == "PEP":
                    # Get PEP energy (0.5 * H2 energy)
                    pep_energy = 0.5 * molecule_energy_dict["H2_g"]

                    total_energy_contribution += (pep_energy - external_potential) * num
                
                # pristine catalysts surface
                elif species == "*":
                    pass
                
                # adsorbed species
                elif species.startswith("*"):
                    pass
                
                # non-adsorbed species (free molecules)
                else:
                    pass
                
                    
            
            
            return total_energy_contribution
        
        
        # Check args
        assert reaction_name in self.reaction_pathway_dict
        
        
        # Unpack necessary inputs
        molecule_energy_dict = self.molecule_energy_dict
        reaction_pathway_dict = self.reaction_pathway_dict[reaction_name]
        external_potential = self.external_potential
        free_energy_linear_relation = self.free_energy_linear_relation
        
        
        # Calculate one linear relation for each reaction step
        result_dict = {}
        for step_index, equation in reaction_pathway_dict.items():
            if step_index != "comment":  # skip comment
                
                # Get terms for products
                products_term = calculate_energy_term(equation["products"], external_potential)
                
                # Get terms for reactants
                reactants_term = calculate_energy_term(equation["reactants"], external_potential) 
                
                # Calculate difference
                
                
                # Add to linear relation
                
         
                
                
                result_dict[int(step_index)] = "test"
        
        
        return result_dict
    
    
    def import_reaction_pathway(self, file):
        """Import reaction pathway file.

        Args:
            file (str): path to reaction pathway file

        Returns:
            dict: reaction pathway dict
            
        """
        # Check args
        assert os.path.exists(file) and file.endswith("json")
        
        # Import reaction pathway file
        with open(file) as f:
            reaction_pathway_dict = json.load(f)
            
        return reaction_pathway_dict 
    
    
    def import_molecule_energy(self, file):
        """Import molecule energy from csv file.

        Args:
            file (str): path to molecule energy file

        Returns:
            dict: dict of molecule-energy pairs
            
        """
        # Check args
        assert os.path.exists(file) and file.endswith(".csv")
        
        # Import molecule energy csv file
        molecule_energy_dict = {}
        with open(file) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for line in reader:
                molecule_energy_dict[line[0]] = float(line[1])
        
        return molecule_energy_dict


# Test area
if __name__ == "__main__":
    # Load configs
    adsorption_energy_path =  "../../0-dataset/label_adsorption_energy"
    reaction_pathway_file = "../data/reaction_pathway.json"
    thermal_correction_file = "../data/corrections_thermal.csv"
    molecule_energy_file = "../data/energy_molecule.csv"
    substrates = ["g-C3N4_is", "nitrogen-graphene_is", "vacant-graphene_is", "C2N_is", "BN_is", "BP_is"]
    adsorbates = ["2-COOH", "3-CO", "4-OCH", "5-OCH2", "6-OCH3", "7-O", "8-OH", "11-H"]
    
    descriptor_x = "3-CO"
    descriptor_y = "8-OH"
    external_potential = 0.17
   
    # Load adsorption energies
    from energy_loader import energyLoader, stack_diff_sub_energy_dict
    from fitting import linear_fitting_with_mixing
    energy_loader = energyLoader()
    energy_loader.load_adsorption_energy(adsorption_energy_path, substrates, adsorbates)

    # Add thermal corrections to adsorption energies
    energy_loader.add_thermal_correction(correction_file=thermal_correction_file)
    
    
    # Perform linear fitting with automatic mixing
    free_energies = stack_diff_sub_energy_dict(energy_loader.free_energy_dict, add_prefix=True)
    
    free_energy_linear_relation = linear_fitting_with_mixing(
        free_energies, 
        descriptor_x, descriptor_y, 
        verbose=False
        )
    
    
    relation = scalingRelations(free_energy_linear_relation,
        molecule_energy_file, reaction_pathway_file, external_potential)
    