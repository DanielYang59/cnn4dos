


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class volcanoPlotter:
    def __init__(self, scaling_relations, x_range, y_range, descriptors, free_energies, *args, **kwargs) -> None:
        # Check args
        assert isinstance(scaling_relations, dict)
        assert isinstance(x_range, tuple)
        assert isinstance(y_range, tuple)
        assert isinstance(descriptors, tuple) and len(descriptors) == 2
        assert isinstance(free_energies, dict)
        
        
        # Update attrib
        self.scaling_relations = scaling_relations
        self.x_range = x_range
        self.y_range = y_range
        self.descriptors = descriptors
        self.free_energies = free_energies
        for key, value in kwargs.items():
            exec(f"self.{key}={value}")

        
        # Generate activity mesh for CO2RR and HER
        her_activity_mesh = self.generate_activity_mesh("HER")
    
    
    def add_colorbar(self, fig, contour, cblabel, ticks):
        """Add colorbar to matplotlib figure.

        Args:
            fig (matplotlib.figure.Figure): _description_
            contour (matplotlib.contour.QuadContourSet): _description_
            cblabel (str): label of colorbar
            ticks (list): list of ticks to display on colorbar
            
        """
        # Create colorbar
        cbar = fig.colorbar(contour, shrink=0.95, aspect=15, ticks=ticks)  # create colorbar (shrink/aspect for the size of the bar)
        
        # Set colorbar format
        cbar.set_label(cblabel, fontsize=35)  # add label to the colorbar
        cbar.ax.tick_params(labelsize=25)  # set tick label size
        cbar.ax.invert_yaxis()  # put the colorbar upside down
        cbar.outline.set_visible(False)  # remove border

        return cbar
        
        
    def add_original_points(self, plt, ):
        # Get x and y list for selected descriptors
        x_list = []
        y_list = []
        name_list = []
        for df in self.free_energies.values():
            x_list.extend(list(df[self.descriptors[0]]))
            y_list.extend(list(df[self.descriptors[1]]))
            name_list.extend(df.index.values)
        
        
        # Compile marker list
        assert len(self.markers) == len(self.free_energies)
        marker_dict = dict(zip(self.free_energies.keys(), self.markers))
        markers = [marker_dict["_".join(name.split("_")[:2])] for name in name_list]
        
        # Add scatters
        for i, _ in enumerate(x_list):
            #print(name_list[i])
            plt.scatter(x_list[i], y_list[i],
                        marker=markers[i],
                        facecolors="#6495ED", edgecolors="black",
                        )
        
        
        # Add labels for selected samples
        
        
        
    
    
    def add_rds_separator_lines(self):
        pass
    
    
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
        self.xx, self.yy = np.meshgrid(
            np.linspace(self.x_range[0], self.x_range[1], density[0]),
            np.linspace(self.y_range[0], self.y_range[1], density[1]),
            )
        
        
        # Calculate energy change for each reaction step
        energy_change_dict = {}
        for step_index, paras in self.scaling_relations[reaction_name].items():
            # Calculate value for each point on mesh
            energy_change_dict[step_index] = self.xx * paras[0] + self.yy * paras[1] + paras[2]
        
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
        stacked_mesh = np.stack(list(activity_mesh.values()), axis=2)
        
        # Get limiting potential mesh
        limiting_potential_mesh = np.amax(stacked_mesh, axis=2)
        
        if return_rds:
            rds_mesh = np.argmax(stacked_mesh, axis=2)
            return limiting_potential_mesh, rds_mesh

        else:
            return limiting_potential_mesh
    
    
    def generate_selectivity_mesh(self, primary_activity_mesh, competing_activity_mesh):
        """Generate selectivity mesh.

        Args:
            primary_activity_mesh (np.ndarray): array of primary reaction limiting potential mesh
            competing_activity_mesh (np.ndarray): array of competing reaction limiting potential mesh

        Notes:
            1. The selectivity mesh is calculated as (primary reaction limiting potential - competing reaction limiting potential). As such LOWER value indicates better selectivity.
        
        Returns:
            np.ndarray: selectivity mesh
            
        """
        # Check args
        assert isinstance(primary_activity_mesh, np.ndarray) and primary_activity_mesh.ndim == 2
        assert isinstance(competing_activity_mesh, np.ndarray) and competing_activity_mesh.ndim == 2
        assert primary_activity_mesh.shape == competing_activity_mesh.shape
        
        # Generate activity mesh for primary reaction
        stacked_primary_mesh = np.stack(list(primary_activity_mesh.values()), axis=2)
        primary_limiting_potential_mesh = np.amax(stacked_primary_mesh, axis=2)  
        
        
        # Generate activity mesh for competing reaction
        stacked_competing_mesh = np.stack(list(primary_activity_mesh.values()), axis=2)
        competing_limiting_potential_mesh = np.amax(stacked_competing_mesh, axis=2)  
        
        
        # Calculate limiting potential difference
        return primary_limiting_potential_mesh - competing_limiting_potential_mesh
    
    
    def plot_limiting_potential(self, reaction_name, show=True, savename="volcano_limiting_potential.png", dpi=300):
        # Check args
        if reaction_name not in self.scaling_relations:
            raise ValueError(f"Cannot find scaling relations for reaction {reaction_name}.")
        
        
        # Generate limiting potential mesh
        # DEBUG: check x/y shape
        print("Check x/y shape!")
        activity_mesh = self.generate_activity_mesh(reaction_name, density=(400, 400))
        
        # Generate limiting potential mesh with rate determining step info
        limiting_potential_mesh, _ = self.generate_limiting_potential_mesh(activity_mesh, return_rds=True)
        
        
        # Create background volcano plot
        fig = plt.figure(figsize=[12, 9])
        
        # Set figure format
        self.set_figure_format(plt)
        
        
        # Create background contour plot
        contour = plt.contourf(self.xx, self.yy, limiting_potential_mesh, 
                               levels=512, cmap="coolwarm_r", # extend="max"
                               )
        
        
        # Add colorbar
        cbar = self.add_colorbar(fig, contour,
                          cblabel="Limiting Potential (V)",
                          ticks=[3, 4, 5, 6],
                          )
        
        
        # Add original data points
        self.add_original_points(plt,)
        
        
        
        # Output figure
        plt.tight_layout()
        plt.savefig(savename, dpi=dpi)
        if show:
            plt.show()
            
            
    def set_figure_format(self, plt):
        # Change font
        font = {'family' : 'sans-serif',
            'sans-serif': 'Helvetica',
            'weight' : 'normal',
            'size'   : 18}
        plt.rc('font', **font)
        

        # Set axis range
        plt.xlim(self.x_range)
        plt.ylim(self.y_range)

        # Set ticks font size
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)

        # Add X/Y axis labels
        # adsorption energy symbol format ref: # DEBUG: confirm adsorption energy symbol format
        plt.xlabel(fr"$\mathit{{G}}_{{\mathit{{ads}}}}\ *{{{self.descriptors[0].split('-')[-1]}}}$ (eV)", fontsize=35)  # x-axis label ("_" for subscript, "\mathit" for Italic)
        plt.ylabel(fr"$\mathit{{G}}_{{\mathit{{ads}}}}\ *{{{self.descriptors[1].split('-')[-1]}}}$ (eV)", fontsize=35)  # y-axis label
        
        return plt
        
        
# Test area
if __name__ == "__main__":
    from calculate_scaling_relations import scalingRelations
    from energy_loader import energyLoader, stack_diff_sub_energy_dict
    from fitting import linear_fitting_with_mixing
    
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
        adsorbate_energy_file,
        reaction_pathway_file,
        external_potential,
        verbose=False,
        ).scaling_relations_dict
    
    
    # Generate volcano plot
    plotter = volcanoPlotter(scaling_relations,
                             x_range=(-5, 1),
                             y_range=(-7, 0),
                             descriptors=("3-CO", "8-OH"),
                             free_energies=energy_loader.free_energy_dict,
                             markers=markers, 
                             )
    
    plotter.plot_limiting_potential(reaction_name="CO2RR_CH4")