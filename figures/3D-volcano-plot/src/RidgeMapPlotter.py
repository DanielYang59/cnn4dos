"""Create 2D/3D ridge plots, with github.com/ColCarroll/ridge_map."""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ridge_map import FontManager, RidgeMap  # type: ignore


class RidgeMapPlotter:
    def __init__(
        self,
        coordinates: Tuple[float, float, float, float],
        font_url: str,
        show_plot: bool = False,
        cmap: str = "ocean",
        interpolate: bool = False,
    ) -> None:
        self.font = FontManager(font_url)
        self.rm = RidgeMap(coordinates, font=self.font.prop)
        self.show_plot = show_plot
        self.cmap = plt.get_cmap(cmap)
        self.interpolate = interpolate

    def get_3D_data(self, num_lines: int) -> None:
        self.values_3D = self.rm.get_elevation_data(num_lines=num_lines)
        self.preprocessed_values_3D = self.rm.preprocess(
            values=self.values_3D, lake_flatness=2, water_ntile=10, vertical_ratio=250
        )

        if self.interpolate:
            self.preprocessed_values_3D = self._interpolate_with_adjacent_means(
                self.preprocessed_values_3D
            )

    def get_2D_data(self, num_lines: int) -> None:
        self.values_2D = self.rm.get_elevation_data(num_lines=num_lines)
        self.preprocessed_values_2D = self.rm.preprocess(
            values=self.values_2D, lake_flatness=2, water_ntile=10, vertical_ratio=250
        )

        if self.interpolate:
            self.preprocessed_values_2D = self._interpolate_with_adjacent_means(
                self.preprocessed_values_2D
            )

    def plot_3D_ridge(self, filename: Path) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        self.rm.plot_map(
            values=self.preprocessed_values_3D,
            label="",
            linewidth=2,
            line_color=self.cmap,
            kind="elevation",
            ax=ax,
        )
        plt.savefig(Path(filename), transparent=True, dpi=600)
        if self.show_plot:
            plt.show()

    def plot_2D_projection(self, filename: Path, alpha: float = 1.0) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(
            np.array(self.preprocessed_values_2D),
            aspect="auto",
            cmap=self.cmap,
            origin="upper",
            alpha=alpha,
        )
        ax.axis("off")
        plt.savefig(Path(filename), transparent=True, dpi=600)
        if self.show_plot:
            plt.show()

    def _interpolate_with_adjacent_means(self, data: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(data)
        df.interpolate(method="linear", axis=1, inplace=True)
        return df.to_numpy()
