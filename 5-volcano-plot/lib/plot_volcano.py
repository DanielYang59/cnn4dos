


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class volcanoPlotter:
    def __init__(self, scaling_relations, x_range, y_range) -> None:
        # Check args
        assert isinstance(scaling_relations, dict)
        assert isinstance(x_range, tuple)
        assert isinstance(y_range, tuple)
        
        
        # Update attrib
        self.scaling_relations = scaling_relations
        self.x_range = x_range
        self.y_range = y_range
        
        
        # Generate activity mesh for CO2RR and HER
        co2rr_activity_mesh = self.generate_activity_mesh("CO2RR_CH4", density=(400, 500))
        her_activity_mesh = self.generate_activity_mesh("HER")
        
        
        # Generate limiting potential mesh with rate determining step info
        limiting_potential_mesh, rds_mesh = self.generate_limiting_potential_mesh(co2rr_activity_mesh, return_rds=True)

        
        # Create CO2RR activity plot
        
        
        # Create CO2RR rate-determining step plot
        
        
        
        # Create CO2RR vs HER selectivity map
        
        
    
    def generate_activity_mesh(self, reaction_name, density=(400, 400)):
        """Generate 2D numpy mesh from scaling relations.

        Args:
            reaction_name (str): name of reaction to generate
            density (tuple, optional): mesh density in (x, y) direction. Defaults to (400, 400).

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        # Check args
        if reaction_name not in self.scaling_relations:
            raise ValueError(f"Cannot find data for reaction {reaction_name}.")
        assert isinstance(density, tuple) and len(density) == 2
        for i in self.x_range:
            assert isinstance(i, (int, float))
        assert self.x_range[0] < self.x_range[1]
        for i in self.y_range:
            assert isinstance(i, (int, float))
        assert self.y_range[0] < self.y_range[1]
        
        
        # Generate a 2D mesh
        xx, yy = np.meshgrid(
            np.linspace(self.x_range[0], self.x_range[1], density[0]),
            np.linspace(self.y_range[0], self.y_range[1], density[1]),
            )
        
        
        # Calculate energy change for each reaction step
        energy_change_dict = {}
        for step_index, paras in self.scaling_relations[reaction_name].items():
            # Calculate value for each point on mesh
            energy_change_dict[step_index] = xx * paras[0] + yy * paras[1] + paras[2]
        
        return energy_change_dict
    
    
    def generate_limiting_potential_mesh(self, activity_mesh, return_rds=True):
        """Generate limiting potential mesh from dict of activity meshes.

        Args:
            activity_mesh (dict): source activity meshes, key is reaction step count, value is activity mesh
            return_rds (bool, optional): return rate determining step mesh. Defaults to True.

        Returns:
            np.ndarray: limiting potential mesh in shape (density_x, density_y)
            
        """
        # Check args
        assert isinstance(activity_mesh, dict)
        
        # Stack all meshes of different steps to shape (density_x, density_y, numSteps)
        stacked_mesh = np.stack(activity_mesh.values(), axis=2)
        
        # Get limiting potential mesh
        limiting_potential_mesh = np.amax(stacked_mesh, axis=2)  # DEBUG: check x/y shape
        
        if return_rds:
            rds_mesh = np.argmax(stacked_mesh, axis=2)
            return limiting_potential_mesh, rds_mesh

        else:
            return limiting_potential_mesh
        
    
    def plot_limiting_potential(self, mesh, show=True, savename="volcano_limiting_potential.png", dpi=300):
        # Check args
        assert isinstance(mesh, np.ndarray) and mesh.ndim == 2
        
        # 

     
    
# Test area
if __name__ == "__main__":
    from calculate_scaling_relations import scalingRelations
    from energy_loader import energyLoader, stack_diff_sub_energy_dict
    from fitting import linear_fitting_with_mixing
    
    adsorption_energy_path = "../../0-dataset/label_adsorption_energy"
    reaction_pathway_file = "../data/reaction_pathway.json"
    thermal_correction_file = "../data/corrections_thermal.csv"
    molecule_energy_file = "../data/energy_molecule.csv"
    non_molecular_adsorbate_energy_file = "../data/energy_adsorbate.csv"
    
    substrates = ["g-C3N4_is", "nitrogen-graphene_is", "vacant-graphene_is", "C2N_is", "BN_is", "BP_is"]
    adsorbates = ["2-COOH", "3-CO", "4-OCH", "5-OCH2", "6-OCH3", "7-O", "8-OH", "11-H"]

    descriptor_x = "3-CO"
    descriptor_y = "8-OH"
    
    external_potential = 0.17 
    
    
    # Load adsorption energies
    energy_loader = energyLoader()
    energy_loader.load_adsorption_energy(adsorption_energy_path, substrates, adsorbates)
    
    # Add thermal corrections to adsorption energies
    energy_loader.add_thermal_correction(correction_file=thermal_correction_file)
    
    
    # Perform linear fitting with automatic mixing
    free_energies = stack_diff_sub_energy_dict(energy_loader.free_energy_dict, add_prefix=True)
    
    free_energy_linear_relation = linear_fitting_with_mixing(
        free_energies,
        descriptor_x, descriptor_y, 
        verbose=False,
        )
    
    # Calculate scaling relations
    scaling_relations = scalingRelations(
        free_energy_linear_relation,
        molecule_energy_file,
        non_molecular_adsorbate_energy_file,
        reaction_pathway_file,
        external_potential,
        verbose=False,
        ).scaling_relations_dict
    
    
    # Generate volcano plot
    plotter = volcanoPlotter(scaling_relations, 
                             x_range = (-5, 2),
                             y_range = (-8, 0),
                             )