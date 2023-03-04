#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class volcanoPlotter:
    def __init__(self, scaling_relations, x_range, y_range, descriptors, adsorption_free_energies, dpi=300, *args, **kwargs):
        # Update attrib
        self.scaling_relations = scaling_relations
        self.x_range = x_range
        self.y_range = y_range
        self.descriptors = descriptors
        self.adsorption_free_energies = adsorption_free_energies
        self.dpi = dpi
        
        for key, value in kwargs.items():
            exec(f"self.{key}={value}")
    
    
    def __add_colorbar(self, fig, contour, cblabel, ticks=None, hide_border=True, invert=True):
        """Add colorbar to matplotlib figure.

        Args:
            fig (matplotlib.figure.Figure): _description_
            contour (matplotlib.contour.QuadContourSet): _description_
            cblabel (str): label of colorbar
            ticks (list): list of ticks to display on colorbar. Defaults to None.
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
        
        
    def __add_markers(self, plt, label_selection="ALL"):
        """Add original data points to volcano plot.

        Args:
            plt (module): plt
            label_selection ((str, list), optional): add labels to selected points or "ALL", select by substrate. Defaults to "ALL".
            
        """
        # Get x and y list for selected descriptors
        x_list = []
        y_list = []
        name_list = []
        for df in self.adsorption_free_energies.values():  
            x_list.extend(list(df[self.descriptors[0]]))
            y_list.extend(list(df[self.descriptors[1]]))
            name_list.extend(df.index.values)
        
        
        # Compile marker list
        assert len(self.markers) == len(self.adsorption_free_energies)
        marker_dict = dict(zip(self.adsorption_free_energies.keys(), self.markers))
        markers = [marker_dict["_".join(name.split("_")[:2])] for name in name_list]
        
        
        # Add scatters
        for i, _ in enumerate(x_list):
            plt.scatter(x_list[i], y_list[i],
                        marker=markers[i],
                        facecolors="#6495ED", edgecolors="black",
                        )
        
        
        # Add labels for selected samples
        for i, name in enumerate(name_list):
            # Compile label names
            if label_selection == "ALL":
                label = name.split("_")[-1].split("-")[-1]         
            else:
                substrate_name = "_".join(name.split("_")[:2])
                
                if substrate_name in label_selection:
                    label = name.split("_")[-1].split("-")[-1]  
                else:
                    label = ""  # would this lead to positioning conflict? DEBUG
                    
            
            # Add annotate
            plt.annotate(label, xy=(x_list[i] + 0.12, y_list[i]),
                         ha="center", va="center",
                         fontsize=12,
                         )
            

    def __generate_free_energy_change_mesh(self, reaction_name, density=(400, 400)):
        """Generate 2D free energy change mesh for plotting.

        Args:
            reaction_name (str): reaction energy
            density (tuple, optional): mesh density in (x, y) directions. Defaults to (400, 400).

        Raises:
            KeyError: if reaction name not in scaling relation dict

        Returns:
            dict: free energy change for each reaction step
            
        """
        # Check for requested reaction name
        if reaction_name not in self.scaling_relations:
            raise KeyError(f"Cannot find entry for reaction {reaction_name}")
        
        
        # Generate 2D mesh
        self.xx, self.yy = np.meshgrid(
            np.linspace(self.x_range[0], self.x_range[1], density[0]),
            np.linspace(self.y_range[0], self.y_range[1], density[1]),
            indexing="ij",
            # outputs are of shape (N, M) for ‘xy’ indexing and (M, N) for ‘ij’ indexing
            )
        
        
        # Get free energy change for each reaction step
        return  {
            step_index: self.xx * paras[0] + self.yy * paras[1] + paras[2]
            # Calculate free energy value for each point on mesh
            for step_index, paras in self.scaling_relations[reaction_name].items()
            }


    def __set_figure_style(self, plt, fig=None):
        """Set figure-wide styles.

        Args:
            plt (module): _
            fig (matplotlib.figure.Figure, optional): _. Defaults to None.

        Returns:
            module: plt
            
        """        
        # Change font
        # font = {'family' : 'sans-serif',
        #     'sans-serif': 'Helvetica',
        #     'weight' : 'normal',
        #     'size'   : 18}
        # plt.rc('font', **font)
        
        
        # Set border thickness
        ax = fig.gca()
        for axis in ["top", "bottom", "left", "right"]:
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
        

    def plot_limiting_potential(self, reaction_name, show=False, label_selection="ALL"):
        """Plot limiting potential volcano of selected reaction.

        Args:
            reaction_name (str): name of reaction to plot
            show (bool, optional): show plot after creation. Defaults to False.
            
        """
        # Generate free energy change mesh for selected reaction
        free_energy_change_mesh = self.__generate_free_energy_change_mesh(reaction_name)
        
        # Stack all meshes of different steps into shape (density_x, density_y, numSteps)
        stacked_mesh = np.stack(list(free_energy_change_mesh.values()), axis=2)
        
        # Generate limiting potential mesh (max of free energy change)
        limiting_potential_mesh = np.amax(stacked_mesh, axis=2)
        
        
        # Create plt object
        mpl.rcParams.update(mpl.rcParamsDefault)  # reset rcParams
        fig = plt.figure(figsize=[12, 9])
        
        
        # Create background contour plot
        contour = plt.contourf(self.xx, self.yy, limiting_potential_mesh,
                               levels=512, cmap="inferno_r",    
                               extend="max",
                               )
        
        
        # Set figure styles
        self.__set_figure_style(plt, fig)
        
        
        # Add colorbar
        cbar = self.__add_colorbar(fig, contour,
                          cblabel="Limiting Potential (V)",
                          ticks=[1, 2, 3, 4, 5],
                          hide_border=False,
                          )
        
        
        # Add markers
        self.__add_markers(plt, 
                           label_selection=label_selection,
                                 )
        
        
        # Save/show figure
        plt.tight_layout()
        plt.savefig(f"limiting_potential_{reaction_name}.png", dpi=self.dpi)
        if show:
            plt.show()
        plt.cla()
        
        
# Test area
if __name__ == "__main__":
    # Set args
    adsorption_energy_path = "../../../0-dataset/label_adsorption_energy"
    reaction_pathway_file = "../../data/reaction_pathway.json"
    thermal_correction_file = "../../data/corrections_thermal.csv"
    adsorbate_energy_file = "../../data/energy_adsorbate.csv"
    
    substrates = ["g-C3N4_is", "nitrogen-graphene_is", "vacant-graphene_is", "C2N_is", "BN_is", "BP_is"]
    adsorbates = ["2-COOH", "3-CO", "4-OCH", "5-OCH2", "6-OCH3", "7-O", "8-OH", "11-H"]

    descriptor_x = "3-CO"
    descriptor_y = "8-OH"
    
    external_potential = 0.17
    
    markers = ["o", "^", "s", "d", "P", "*"]
    
    
    # Loading adsorption energy
    from .dataLoader import dataLoader
    loader = dataLoader()
    loader.load_adsorption_energy(adsorption_energy_path, substrates, adsorbates)
    
    loader.calculate_adsorption_free_energy(correction_file="../../data/corrections_thermal.csv")
    
    # Calculate adsorption energy linear scaling relations
    from .scalingRelation import scalingRelation
    calculator = scalingRelation(adsorption_energy_dict=loader.adsorption_free_energy, descriptors=("3-CO", "8-OH"), mixing_ratios="AUTO", verbose=False, remove_ads_prefix=True) 

    # Calculate reaction energy scaling relations calculator
    from .reactionCalculator import reactionCalculator
    reaction_calculator = reactionCalculator(
        adsorption_energy_scaling_relation=calculator.fitting_paras,
        adsorbate_energy_file="../../data/energy_adsorbate.csv",
        reaction_pathway_file="../../data/reaction_pathway.json",
        external_potential=0.17,
        )
    
    
    reaction_scaling_relations = {
        "CO2RR_CH4":reaction_calculator.calculate_reaction_scaling_relations(name="CO2RR_CH4"),
        "HER":reaction_calculator.calculate_reaction_scaling_relations(name="HER")
                         }
    
    
    # Initialize volcano plotter
    plotter = volcanoPlotter(reaction_scaling_relations,
                             x_range=(-5, 0.5),
                             y_range=(-6.5, 0),
                             descriptors=("3-CO", "8-OH"),
                             adsorption_free_energies=loader.adsorption_free_energy,
                             markers=markers,
                             )
    
    # Generate limiting potential volcano plot
    plotter.plot_limiting_potential(reaction_name="CO2RR_CH4", show=True,
                                    label_selection=["g-C3N4_is", "nitrogen-graphene_is", "vacant-graphene_is",]
                                    )


    # Generate rate determining step plot
    # plotter.plot_rds(reaction_name="CO2RR_CH4")
    
    
    # Generate selectivity volcano plot
    
    