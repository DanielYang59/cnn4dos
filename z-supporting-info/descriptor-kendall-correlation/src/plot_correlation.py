#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from .heatmap_revised import corrplot  # pip install heatmapz


def plot_correlation(dataset, method, savename, show=True):
    # Calculate correlation map
    assert isinstance(dataset, pd.DataFrame)
    corr_data = dataset.corr(method=method, numeric_only=True)
    
    
    # Generate correlation plot
    sns.set(color_codes=True, font_scale=1.2)
    plt.figure(figsize=(8, 8))
    corrplot(
        corr_data,
        size_scale=150, marker="s",
    )


    # Save and show plot
    plt.savefig(savename, bbox_inches="tight", dpi=600)
    if show:
        plt.show()
