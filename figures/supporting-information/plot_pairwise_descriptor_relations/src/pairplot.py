"""Plot pairwise plots."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]


class PairPlot:
    def __init__(self, data) -> None:
        assert isinstance(data, pd.DataFrame)
        self.data = data

    def __preprocess_data(self, descriptors):
        """Preprocess data for pairwise plot, including:
               remove metal/substrate name columns
               remove non-numerical columns
               drop entry with missing value (NaN)
               select only specific descriptors

        Args:
            descriptors (list): descriptors to plot (most
                correlated determined by Kendall coefficients)

        """
        # Drop metal/substrate name columns for plotting
        self.data.drop(labels=["metal", "substrate"], axis=1, inplace=True)

        # Drop non-numerical columns
        self.data = self.data._get_numeric_data()

        # Drop lines with missing value
        self.data.dropna(axis=0, inplace=True)

        # Take selected descriptors only
        self.data = self.data.loc[:, descriptors]

    def plot_pairplot(
        self, descriptors, descriptor_symbol_dict, savename, show=False
    ) -> None:
        """Generate pairwise relations plot.

        Args:
            descriptors (list): most correlated descriptors determined
                by Kendall correlation coefficient
            descriptor_symbol_dict(dict):
            savename (Path): name of figure
            show (bool, optional): show figure during plotting.
                Defaults to True.

        Notes:
            Ref: https://seaborn.pydata.org/generated/seaborn.pairplot.html

        """
        # Preprocess datasheet
        self.__preprocess_data(descriptors=descriptors)

        # Rename datasheet columns
        self.data = self.data.rename(columns=descriptor_symbol_dict)

        # Generate pairwise relation plot
        with sns.plotting_context(rc={"axes.labelsize": 32, "axes.linewidth": 2.5}):
            sns.pairplot(
                data=self.data,
                corner=True,
                diag_kind="kde",
                # diag_kws= {'color': '#82ad32'},  # set diagonal color
                plot_kws={"s": 100, "color": "#9925be", "alpha": 0.8},
            )

        # Save and show figure
        plt.savefig(savename, dpi=300)
        if show:
            plt.show()
