"""Plot the correlation map."""


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

from .heatmap_revised import corrplot  # pip install heatmapz

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]


def plot_correlation(dataset, method, savename, show=True, verbose=False):
    """Plot correlation map.

    Args:
        dataset (pd.DataFrame): dataset to be plotted
        method (str): method for correlation calculation
        savename (Path): savename of figure
        show (bool, optional): show figure during plotting. Defaults to True.
        verbose (bool, optional): print correlation data. Defaults to False.

    """
    # Calculate correlation map
    assert isinstance(dataset, pd.DataFrame)
    corr_data = dataset.corr(method=method, numeric_only=True)

    # Print correlation data if required
    if verbose:
        print(f'"Absolute value" of {method} correlation coefficients:')
        print(corr_data.iloc[:, 0].abs().sort_values(ascending=False))

    # Generate correlation plot
    sns.set(color_codes=True, font_scale=1.2)
    plt.figure(figsize=(8, 8))
    corrplot(
        corr_data,
        size_scale=150,
        marker="s",
    )

    # Save and show plot
    plt.savefig(savename, bbox_inches="tight", dpi=600)
    if show:
        plt.show()
