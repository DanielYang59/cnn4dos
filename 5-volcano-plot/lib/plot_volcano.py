


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class volcanoPlotter:
    def __init__(self, scaling_relations, x_range, y_range, descriptors, free_energies, *args, **kwargs) -> None:
        """Volcano plot plotter based on scaling relations.

        Args:
            scaling_relations (dict): _description_
            x_range (tuple): _description_
            y_range (tuple): _description_
            descriptors (tuple): (descriptor for x-axis, descriptor for y-axis)
            free_energies (dict): _description_
            
        """
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

    
    def add_colorbar(self, fig, contour, cblabel, ticks, hide_border=True, invert=True):
        """Add colorbar to matplotlib figure.

        Args:
            fig (matplotlib.figure.Figure): _description_
            contour (matplotlib.contour.QuadContourSet): _description_
            cblabel (str): label of colorbar
            ticks (list): list of ticks to display on colorbar
            hide_border (bool, optional): hide colorbar border. Defaults to True.
            invert (bool, optional): invert colorbar. Defaults to True.
            
        """
        # Create colorbar
        cbar = fig.colorbar(contour, shrink=0.95, aspect=15, ticks=ticks)  # create colorbar (shrink/aspect for the size of the bar)
        
        # Set colorbar format
        cbar.set_label(cblabel, fontsize=35)  # add label to the colorbar
        cbar.ax.tick_params(labelsize=25, length=5, width=2.5)  # set tick label size and tick style
        if invert:
            cbar.ax.invert_yaxis()  # put the colorbar upside down
        if hide_border:
            cbar.outline.set_visible(False)  # remove border
        else:
            cbar.outline.set_linewidth(1.5)  # set border thickness

        return cbar
        
        
    def add_markers(self, plt, label_selection="ALL"):
        """Add original data points to volcano plot.

        Args:
            plt (module): plt
            label_selection ((str, list), optional): add labels to selected points or ALL, select by substrate. Defaults to "ALL".
            
        """
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
        # for i, label in enumerate(marker_label_list):
        # plt.annotate(label, (x_data_list[i] - 0.1, y_data_list[i]), # Offset by some value
        #              ha='center', va='center',
        #              fontsize=16,
        #              )
        
        
    def add_rds_separator_lines(self):
        pass
    
    
    def generate_activity_mesh(self, reaction_name, density=(400, 400)):
        """Generate 2D numpy mesh from scaling relations.

        Args:
            reaction_name (str): name of reaction to generate
            density (tuple, optional): mesh density in (x, y) direction. Defaults to (400, 400).

        Raises:
            ValueError: if can not find selected reaction name

        Returns:
            dict: dict of scaling relations, where step index is key and scaling relations are values
            
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
            indexing="ij",  # outputs are of shape (N, M) for ‘xy’ indexing and (M, N) for ‘ij’ indexing
            )
        

        # Calculate energy change for each reaction step
        return  {
            step_index: self.xx * paras[0] + self.yy * paras[1] + paras[2]
            # Calculate value for each point on mesh
            for step_index, paras in self.scaling_relations[reaction_name].items()
            }
    
    
    def generate_limiting_potential_mesh(self, activity_mesh, return_rds=True, ref_to_min=False):
        """Generate limiting potential mesh from dict of activity meshes.

        Args:
            activity_mesh (dict): source activity meshes, key is reaction step count, value is activity mesh
            return_rds (bool, optional): return rate determining step mesh. Defaults to True.
            ref_to_min (bool, optional): use min limiting potential as reference. Defaults to True. 

        Notes:
            1. Rate determining step starts from "1".
        
        Returns:
            np.ndarray: limiting potential mesh in shape (density_x, density_y)
            
        """
        # Check args
        assert isinstance(activity_mesh, dict)
        
        # Stack all meshes of different steps to shape (density_x, density_y, numSteps)
        stacked_mesh = np.stack(list(activity_mesh.values()), axis=2)
        
        # Get limiting potential mesh
        limiting_potential_mesh = np.amax(stacked_mesh, axis=2)
        
        if ref_to_min:
            limiting_potential_mesh -= limiting_potential_mesh.min()
        
        if return_rds:
            rds_mesh = np.argmax(stacked_mesh, axis=2)
            rds_mesh += 1  # offset RDS value (step starts from 1)
            
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
    
    
    def plot_limiting_potential(self, reaction_name, show=True, savename="volcano_limiting_potential.png", dpi=300, ref_to_min=False):
        # Check args
        if reaction_name not in self.scaling_relations:
            raise ValueError(f"Cannot find scaling relations for reaction {reaction_name}.")
        
        
        # Generate limiting potential mesh
        activity_mesh = self.generate_activity_mesh(reaction_name, density=(400, 500))
        
        # Generate limiting potential mesh with rate determining step info
        limiting_potential_mesh, _ = self.generate_limiting_potential_mesh(activity_mesh, return_rds=True, ref_to_min=ref_to_min)
        
        
        # Create background volcano plot
        mpl.rcParams.update(mpl.rcParamsDefault)  # reset rcParams
        fig = plt.figure(figsize=[12, 9])
        
        
        # Set figure format
        self.set_figure_format(plt, fig)
        
        # Add X/Y axis labels
        # adsorption energy symbol format ref: # DEBUG: confirm adsorption energy symbol format
        plt.xlabel(fr"$\mathit{{G}}_{{\mathit{{ads}}}}\ *{{{self.descriptors[0].split('-')[-1]}}}$ (eV)", fontsize=35)  # x-axis label ("_" for subscript, "\mathit" for Italic)
        plt.ylabel(fr"$\mathit{{G}}_{{\mathit{{ads}}}}\ *{{{self.descriptors[1].split('-')[-1]}}}$ (eV)", fontsize=35)  # y-axis label
        
        
        # Create background contour plot
        contour = plt.contourf(self.xx, self.yy, limiting_potential_mesh, 
                               levels=512, cmap="plasma_r", 
                               extend="max", 
                               )
        
        
        # Add colorbar
        cbar = self.add_colorbar(fig, contour,
                          cblabel="ΔLimiting Potential (V)" if ref_to_min else "Limiting Potential (V)",
                          ticks=[3, 4, 5],
                          hide_border=False,
                          )
        
        
        # Add original data points
        self.add_markers(plt, 
                                 # label_selection=
                                 )
        
        
        # Add RDS steps separator lines
        self.add_rds_separator_lines()
        
        
        # Output figure
        plt.tight_layout()
        plt.savefig(savename, dpi=dpi)
        if show:
            plt.show()
        plt.cla()
    
    
    def plot_rds(self, reaction_name, show=True, savename="volcano_RDS.png", dpi=150):
        # Check args
        if reaction_name not in self.scaling_relations:
            raise ValueError(f"Cannot find scaling relations for reaction {reaction_name}.")
        
        
        # Generate limiting potential mesh
        activity_mesh = self.generate_activity_mesh(reaction_name, density=(400, 500))
        
        # Generate limiting potential mesh with rate determining step info
        _, rds_mesh = self.generate_limiting_potential_mesh(activity_mesh, return_rds=True, ref_to_min=False)
        
        
        # Create background volcano plot
        mpl.rcParams.update(mpl.rcParamsDefault)  # reset rcParams
        fig = plt.figure(figsize=[12, 9])
        
        
        # Set figure format
        self.set_figure_format(plt, fig)
        
        # Add X/Y axis labels
        # adsorption energy symbol format ref:
        plt.xlabel(fr"$\mathit{{G}}_{{\mathit{{ads}}}}\ *{{{self.descriptors[0].split('-')[-1]}}}$ (eV)", fontsize=35)  # x-axis label ("_" for subscript, "\mathit" for Italic)
        plt.ylabel(fr"$\mathit{{G}}_{{\mathit{{ads}}}}\ *{{{self.descriptors[1].split('-')[-1]}}}$ (eV)", fontsize=35)  # y-axis label
        
        
        # Create background contour plot
        contour = plt.contourf(self.xx, self.yy, rds_mesh, 
                            levels=128, cmap="rainbow", 
                            )
        
        
        # Add colorbar
        cbar = self.add_colorbar(fig, contour,
                        cblabel="Rate Determining Step",
                        ticks=range(1, len(activity_mesh) + 1),
                        hide_border=False,
                        )
        
        
        # Add original data points
        self.add_markers(plt)
        
        
        # Output figure
        plt.tight_layout()
        plt.savefig(savename, dpi=dpi)
        if show:
            plt.show()
        plt.cla()
                    
            
    def set_figure_format(self, plt, fig=None):
        """Set figure-wide styles.

        Args:
            plt (module): _
            fig (matplotlib.figure.Figure, optional): _. Defaults to None.

        Returns:
            module: plt
            
        """        
        # Change font
        font = {'family' : 'sans-serif',
            'sans-serif': 'Helvetica',
            'weight' : 'normal',
            'size'   : 18}
        plt.rc('font', **font)
        
        
        # Set border thickness
        ax = fig.gca()
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.75)
        
        # Set tick thickness
        ax.xaxis.set_tick_params(length=5, width=2.5)
        ax.yaxis.set_tick_params(length=5, width=2.5)
        
        # Set ticks font size
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        
        
        # Set axis range
        plt.xlim(self.x_range)
        plt.ylim(self.y_range)
      
        return plt, fig
        
        
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
                             x_range=(-5, 0.5),
                             y_range=(-6.5, 0),
                             descriptors=("3-CO", "8-OH"),
                             free_energies=energy_loader.free_energy_dict,
                             markers=markers, 
                             )
    
    # plotter.plot_limiting_potential(reaction_name="CO2RR_CH4", ref_to_min=False)

    
    plotter.plot_rds(reaction_name="CO2RR_CH4")
    