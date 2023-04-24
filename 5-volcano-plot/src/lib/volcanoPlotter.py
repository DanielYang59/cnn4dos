#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import warnings


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


    def __add_colorbar(self, fig, contour, cblabel, ticks=None, hide_border=True, invert=False):
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

        def calculate_scatter_alpha(x_list, y_list):
            """Calculate transparency(alpha) based on limiting potential difference.

            Args:
                x_list (list): x coordinate list
                y_list (list): y coordinate list

            Returns:
                np.ndarray: alpha array

            """
            # Locate the maximum
            max_index = np.unravel_index(np.argmax(self.limiting_potential_mesh), self.limiting_potential_mesh.shape)
            x_coord, y_coord = self.x[max_index[1]], self.y[max_index[0]]


            # Calculate distance to maximum
            distances = []
            for i, _ in enumerate(x_list):
                distances.append(math.hypot(x_coord - x_list[i], y_coord - y_list[i]))


            # Scale distance to get alpha values
            alphas = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

            return 1 - alphas * 0.75  # closer to max, less transparent


        # Get x and y list for selected descriptors
        x_list = []
        y_list = []
        name_list = []
        for df in self.adsorption_free_energies.values():
            x_list.extend(list(df[self.descriptors[0]]))
            y_list.extend(list(df[self.descriptors[1]]))
            name_list.extend(df.index.values)
        x_list = np.array(x_list)
        y_list = np.array(y_list)


        # Compile marker list
        assert len(self.markers) == len(self.adsorption_free_energies)
        marker_dict = dict(zip(self.adsorption_free_energies.keys(), self.markers))
        markers = [marker_dict["_".join(name.split("_")[:2])] for name in name_list]


        # Calculate alpha (transparency) based on distance to maximum value
        alphas = calculate_scatter_alpha(x_list, y_list)


        # Add scatters
        for i, _ in enumerate(x_list):
            plt.scatter(x_list[i], y_list[i],
                        marker=markers[i],
                        facecolors="#6495ED", edgecolors="black",
                        )


        # Add legend manually
        ## Ref: https://stackoverflow.com/questions/39500265/how-to-manually-create-a-legend
        warnings.warn("Legend is manually inserted.")

        marker_circle = Line2D([], [], color='#1f77b4', marker='o', linestyle='None', markersize=10)
        marker_triangle = Line2D([], [], color='#1f77b4', marker='^', linestyle='None', markersize=10)
        marker_square = Line2D([], [], color='#1f77b4', marker='s', linestyle='None', markersize=10)
        marker_diamond = Line2D([], [], color='#1f77b4', marker='D', linestyle='None', markersize=10)
        marker_plus = Line2D([], [], color='#1f77b4', marker='P', linestyle='None', markersize=10)
        marker_star = Line2D([], [], color='#1f77b4', marker='*', linestyle='None', markersize=10)

        plt.legend(
            bbox_to_anchor=(0.45, 0.9),
            framealpha=0.3,  # alpha of background color
            fontsize=13,
            handles=[marker_circle, marker_triangle, marker_square, marker_diamond, marker_plus, marker_star],
            labels=[r'g-C$_{3}$N$_{4}$', "nitrogen-doped graphene", "graphene with dual-vacancy", r'C$_{2}$N', "boron nitride ", "black phosphorous"],
            )\
            .get_frame().set_boxstyle('Round', pad=0.2, rounding_size=2)  # use round cornered (ref: https://stackoverflow.com/questions/62972429/how-to-change-legend-edges-from-round-to-sharp-corners)



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
                    label = ""  # WARNING: this might lead to positioning conflict?


            # Add annotate
            plt.annotate(label, xy=(x_list[i] + 0.12, y_list[i]),
                         ha="center", va="center",
                         fontsize=12,
                         alpha=alphas[i],
                         )


    def __add_rds_separator(self, plt):
        """Add separator lines and RDS index for each rate determining step area.

        Args:
            plt (_type_): _description_

        Note:
            DEBUG: temporary manual-method, need improvement

        """
        warnings.warn("RDS separator manually created.")

        # Add separator lines
        line_width = 3
        line_color = "#8F00FF"
        line_style = (0, (1, 1.5))  # (offset, (on_off_seq)) (ref: https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html)

        plt.plot([-5.00, -3.40, -1.23, -0.96, -2.08], [-5.23, -4.92, -2.43, -1.43, 0], linewidth=line_width, color=line_color, linestyle=line_style, dash_capstyle="round")
        plt.plot([-3.40, -3.10], [-4.92, -6.50], linewidth=line_width, color=line_color, linestyle=line_style, dash_capstyle="round")
        plt.plot([-1.23, 0.5], [-2.43, -2.38], linewidth=line_width, color=line_color, linestyle=line_style, dash_capstyle="round")
        plt.plot([-0.96, 0.5], [-1.43, -0.71], linewidth=line_width, color=line_color, linestyle=line_style, dash_capstyle="round")


        # Add RDS indexes (starts from 1)
        font_size = 30

        plt.text(-4.8, -6.2, "I", fontsize=font_size)
        plt.text(-4.8, -0.5, "II", fontsize=font_size)
        plt.text(0, -6.2, "III", fontsize=font_size)
        plt.text(0.25, -1.7, "V", fontsize=font_size)
        plt.text(-0.75, -0.4, "VII", fontsize=font_size)


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
        self.x = np.linspace(self.x_range[0], self.x_range[1], density[0])
        self.y = np.linspace(self.y_range[0], self.y_range[1], density[1])
        self.xx, self.yy = np.meshgrid(
            self.x,
            self.y,
            )


        # Get free energy change for each reaction step
        return  {
            step_index: self.xx * paras[0] + self.yy * paras[1] + paras[2]
            # Calculate free energy value for each point on mesh
            for step_index, paras in self.scaling_relations[reaction_name].items()
            }


    def __generate_limiting_potential_and_RDS_mesh(self, free_energy_change_mesh, reaction_name=None, show_best=True):
        """Generate limiting potential mesh from free energy change meshes.

        Args:
            free_energy_change_mesh (dict): free energy mesh in (x, y, numSteps)
            reaction_name (str): reaction name

        Returns:
            np.ndarray: 2D limiting potential mesh

        """
        # Stack all meshes of different steps into shape (density_x, density_y, numSteps)
        stacked_mesh = np.stack(list(free_energy_change_mesh.values()), axis=2)


        # Generate limiting potential mesh (negate max of free energy change)
        limiting_potential_mesh = -np.amax(stacked_mesh, axis=2)


        # Generate limiting potential mesh (index of min limiting potential)
        rds_mesh = np.argmin(stacked_mesh, axis=2)


        # Find x/y coordinates of maximum (predicted best limiting potential)
        max_index = np.unravel_index(np.argmax(limiting_potential_mesh), limiting_potential_mesh.shape)
        x_coord, y_coord = self.x[max_index[1]], self.y[max_index[0]]

        if show_best:
            print(f"Limiting potential of best {reaction_name} catalysts is {np.max(limiting_potential_mesh):.4f} V, at X {x_coord:.4f} eV, Y {y_coord:.4f} eV.")

        return limiting_potential_mesh, rds_mesh


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


    def plot_limiting_potential(self, reaction_name, show=False, label_selection="ALL", savename="limiting_potential.png"):
        """Plot limiting potential volcano of selected reaction.

        Args:
            reaction_name (str): name of reaction to plot
            show (bool, optional): show plot after creation. Defaults to False.

        """
        # Generate free energy change mesh for selected reaction
        free_energy_change_mesh = self.__generate_free_energy_change_mesh(reaction_name)


        # Generate limiting potential mesh
        self.limiting_potential_mesh, _ = self.__generate_limiting_potential_and_RDS_mesh(free_energy_change_mesh, reaction_name)


        # Create plt object
        mpl.rcParams.update(mpl.rcParamsDefault)  # reset rcParams to default
        fig = plt.figure(figsize=[12, 9])


        # Add x/y axis labels
        plt.xlabel(fr"$\mathit{{G}}_{{\mathregular{{ads}}}}$ *{self.descriptors[0].split('-')[-1]} (eV)", fontsize=35)  # x-axis label ("_" for subscript, "\mathit" for Italic)
        plt.ylabel(fr"$\mathit{{G}}_{{\mathregular{{ads}}}}$ *{self.descriptors[1].split('-')[-1]} (eV)", fontsize=35)  # y-axis label


        # Create background contour plot
        contour = plt.contourf(self.xx, self.yy, self.limiting_potential_mesh,
                               levels=512, cmap="coolwarm",
                               # extend="min",
                               )


        # Set figure styles
        self.__set_figure_style(plt, fig)


        # Add colorbar
        cbar = self.__add_colorbar(fig, contour,
                          cblabel="Limiting Potential (V)",
                          ticks=[-1.1, -2, -3, -4, -5],
                          hide_border=False,
                          invert=False
                          )


        # Add markers
        self.__add_markers(plt,
                           label_selection=label_selection,
                                 )


        # Add RDS separator
        self.__add_rds_separator(plt)


        # Save/show figure
        plt.tight_layout()
        plt.savefig(savename, dpi=self.dpi)
        if show:
            plt.show()
        plt.cla()


    def plot_rds(self, reaction_name, show=False, savename="rds.png"):
        """Plot rate determining step volcano of selected reaction.

        Args:
            reaction_name (str): name of reaction to plot
            show (bool, optional): show plot after creation. Defaults to False.

        """
        # Generate free energy change mesh for selected reaction
        free_energy_change_mesh = self.__generate_free_energy_change_mesh(reaction_name)


        # Generate RDS mesh
        _, rds_mesh = self.__generate_limiting_potential_and_RDS_mesh(free_energy_change_mesh, show_best=False)
        rds_mesh += 1  # reaction step index starts from 1


        # Create plt object
        mpl.rcParams.update(mpl.rcParamsDefault)  # reset rcParams
        fig = plt.figure(figsize=[12, 9])


        # Add x/y axis labels
        plt.xlabel(fr"$\mathit{{G}}_{{\mathit{{ads}}}}\ *{{{self.descriptors[0].split('-')[-1]}}}$ (eV)", fontsize=35)  # x-axis label ("_" for subscript, "\mathit" for Italic)
        plt.ylabel(fr"$\mathit{{G}}_{{\mathit{{ads}}}}\ *{{{self.descriptors[1].split('-')[-1]}}}$ (eV)", fontsize=35)  # y-axis label

        # Set figure styles
        self.__set_figure_style(plt, fig)


        # Create background contour plot
        cmap = ListedColormap(["blue", "green", "yellow", "orange", "red", "purple", "brown", "gray"])

        bounds = [1, 2, 3, 4, 5, 6, 7, 8]
        norm = BoundaryNorm(bounds, cmap.N)

        contour = plt.contourf(self.xx, self.yy, rds_mesh,
                               levels=10, cmap=cmap,
                               )

        # Add discrete colorbar
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=bounds, boundaries=bounds)

        # Save/show figure
        plt.tight_layout()
        plt.savefig(savename, dpi=self.dpi)
        if show:
            plt.show()
        plt.cla()


    def plot_selectivity(self, reaction_names, show=False, label_selection="ALL", savename="selectivity.png"):
        """Plot selectivity volcano of selected reaction.

        Args:
            reaction_name (dict): {"main": main_reaction_name, "comp": competing_reaction_name}
            show (bool, optional): show plot after creation. Defaults to False.


        Notes:
            1. The selectivity mesh is calculated as the (UL_main - UL_competing), where the UL is the limiting potential in eV. This means "more positive value in the volcano plot indicates better selectivity".

        """
        # Generate free energy change mesh for main and competing reactions
        free_energy_change_mesh_main = self.__generate_free_energy_change_mesh(reaction_names["main"])
        free_energy_change_mesh_comp = self.__generate_free_energy_change_mesh(reaction_names["comp"])


        # Generate limiting potential mesh for each reaction
        lim_potential_mesh_main, _ = self.__generate_limiting_potential_and_RDS_mesh(free_energy_change_mesh_main, show_best=False)
        lim_potential_mesh_comp, _ = self.__generate_limiting_potential_and_RDS_mesh(free_energy_change_mesh_comp, show_best=False)


        # Calculate selectivity mesh
        selectivity_mesh = lim_potential_mesh_main - lim_potential_mesh_comp


        # Create plt object
        mpl.rcParams.update(mpl.rcParamsDefault)  # reset rcParams
        fig = plt.figure(figsize=[12, 9])


        # Add x/y axis labels
        plt.xlabel(fr"$\mathit{{G}}_{{\mathregular{{ads}}}}$ *{self.descriptors[0].split('-')[-1]} (eV)", fontsize=35)  # x-axis label ("_" for subscript, "\mathit" for Italic)
        plt.ylabel(fr"$\mathit{{G}}_{{\mathregular{{ads}}}}$ *{self.descriptors[1].split('-')[-1]} (eV)", fontsize=35)  # y-axis label


        # Create background contour plot
        contour = plt.contourf(self.xx, self.yy, selectivity_mesh,
                               levels=512, cmap="coolwarm",
                               # extend="min",
                               )

        # # Add limitint potential difference == 0 line
        # contour_line = plt.contour(self.xx, self.yy, selectivity_mesh, levels=[0],
        #                            color="black", linestyle="-", linewidth=2,
        #                            )
        # plt.clabel(contour_line, fmt='%2.1d', colors='k', fontsize=14)  # contour line labels


        # Set figure styles
        self.__set_figure_style(plt, fig)


        # Add colorbar
        cbar = self.__add_colorbar(fig, contour,
                          cblabel="ΔLimiting Potential (V)",
                          ticks=[-2, -1, 0, 1, 2],
                          hide_border=False,
                          )


        # Add markers
        self.__add_markers(plt,
                           label_selection=label_selection,
                                 )


        # Save/show figure
        plt.tight_layout()
        plt.savefig(savename, dpi=self.dpi)
        if show:
            plt.show()
        plt.cla()


# Test area
if __name__ == "__main__":
    from pathlib import Path
    # Set args
    adsorption_energy_path = Path("../../../0-dataset/label_adsorption_energy")
    reaction_pathway_file = Path("../../data/reaction_pathway.json")
    thermal_correction_file = Path("../../data/corrections_thermal.csv")
    adsorbate_energy_file = Path("../../data/energy_adsorbate.csv")

    substrates = ["g-C3N4_is", "nitrogen-graphene_is", "vacant-graphene_is", "C2N_is", "BN_is", "BP_is"]
    adsorbates = ["2-COOH", "3-CO", "4-CHO", "5-CH2O", "6-OCH3", "7-O", "8-OH", "11-H"]

    descriptor_x = "3-CO"
    descriptor_y = "8-OH"

    external_potential = 0.17

    markers = ["o", "^", "s", "d", "P", "*"]


    # Loading adsorption energy
    from .dataLoader import dataLoader
    loader = dataLoader()
    loader.load_adsorption_energy(adsorption_energy_path, substrates, adsorbates)

    loader.calculate_adsorption_free_energy(correction_file=Path("../../data/corrections_thermal.csv"))

    # Calculate adsorption energy linear scaling relations
    from .scalingRelation import scalingRelation
    calculator = scalingRelation(adsorption_energy_dict=loader.adsorption_free_energy, descriptors=("3-CO", "8-OH"), mixing_ratios="AUTO", verbose=False, remove_ads_prefix=True)

    # Calculate reaction energy scaling relations calculator
    from .reactionCalculator import reactionCalculator
    reaction_calculator = reactionCalculator(
        adsorption_energy_scaling_relation=calculator.fitting_paras,
        adsorbate_energy_file=Path("../../data/energy_adsorbate.csv"),
        reaction_pathway_file=Path("../../data/reaction_pathway.json"),
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
