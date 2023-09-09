#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ridge_map import FontManager, RidgeMap
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class RidgeMapPlotter:
    """
    A class for plotting 3D and 2D ridge maps using ridge_map library.

    Attributes:
        font (FontManager): Font manager object.
        rm (RidgeMap): Ridge map object.
        cmap (matplotlib.colors.Colormap): The colormap used for plotting.
    """
    def __init__(self, coordinates, font_url, show_plot=False, cmap='ocean'):
        """
        Initialize a RidgeMapPlotter object.

        Args:
            coordinates (tuple): Tuple of coordinates (lon_min, lat_min, lon_max, lat_max).
            font_url (str): URL of the font to use.
            cmap (str): The colormap to use for plotting. Defaults to 'ocean'.
        """
        self.font = FontManager(font_url)
        self.rm = RidgeMap(coordinates, font=self.font.prop)
        self.show_plot = show_plot
        self.cmap = plt.get_cmap(cmap)

    def get_3D_data(self, num_lines):
        """
        Generate data for 3D plot.

        Args:
            num_lines (int): Number of lines for the 3D plot.
        """
        self.values_3D = self.rm.get_elevation_data(num_lines=num_lines)
        self.preprocessed_values_3D = self.rm.preprocess(values=self.values_3D, lake_flatness=2, water_ntile=10, vertical_ratio=250)

    def get_2D_data(self, num_lines):
        """
        Generate data for 2D plot.

        Args:
            num_lines (int): Number of lines for the 2D plot.
        """
        self.values_2D = self.rm.get_elevation_data(num_lines=num_lines)
        self.preprocessed_values_2D = self.rm.preprocess(values=self.values_2D, lake_flatness=2, water_ntile=10, vertical_ratio=250)

    def plot_3D_ridge(self, filename="Hawaii_3D_Ridge_Map.png"):
        fig, ax = plt.subplots(figsize=(10, 8))
        self.rm.plot_map(values=self.preprocessed_values_3D,
                         label="",
                         linewidth=2,
                         line_color=self.cmap,
                         kind='elevation',
                         ax=ax
                         )
        plt.savefig(Path(filename), dpi=600)
        if self.show_plot:
            plt.show()


    def plot_2D_projection(self, filename="Hawaii_2D_Projection.png", alpha=1.0):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(np.array(self.preprocessed_values_2D), aspect='auto', cmap=self.cmap, origin='upper', alpha=alpha)
        ax.axis('off')
        plt.savefig(Path(filename), dpi=600)
        if self.show_plot:
            plt.show()
