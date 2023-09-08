#!/bin/usr/python3
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
        values (ndarray): Elevation data.
        preprocessed_values (ndarray): Preprocessed elevation data.
    """

    def __init__(self, coordinates, font_url, cmap="ocean"):
        """
        Initialize a RidgeMapPlotter object.

        Args:
            coordinates (tuple): Tuple of coordinates (lon_min, lat_min, lon_max, lat_max).
            font_url (str): URL of the font to use.
            cmap (str): The colormap to use for plotting. Defaults to 'ocean'.
        """
        self.font = FontManager(font_url)
        self.rm = RidgeMap(coordinates, font=self.font.prop)
        self.values = self.rm.get_elevation_data(num_lines=150)
        self.preprocessed_values = self.rm.preprocess(values=self.values, lake_flatness=2, water_ntile=10, vertical_ratio=250)
        self.cmap = plt.get_cmap(cmap)

    def plot_3D_ridge(self, filename="3D_volcano.png"):
        """
        Plot a 3D ridge map and save it to a file.

        Args:
            filename (str): Name of the file to save the plot. Defaults to "3D_volcano.png".
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        self.rm.plot_map(values=self.preprocessed_values,
                         label="",  # hide label
                         linewidth=2,
                         line_color=self.cmap,
                         kind='elevation',
                         ax=ax)
        plt.savefig(Path(filename), dpi=600)
        plt.show()

    def plot_2D_projection(self, filename="2D_volcano.png"):
        """
        Plot a 2D projection of the ridge map and save it to a file.

        Args:
            filename (str): Name of the file to save the plot. Defaults to "2D_volcano.png".
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(np.array(self.preprocessed_values), aspect="auto", cmap=self.cmap, origin="upper")
        ax.axis('off')  # hide axes for 2D projection
        plt.savefig(Path(filename), dpi=600)
        plt.show()
