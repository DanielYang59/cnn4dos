#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_mixing_ratio(x, y, savename):
    """Plot mixing ratio for hybrid descriptor test.

    Args:
        x (list): x
        y (list): y
        savename (Path): savename of figure

    """

    assert len(x) == len(y)

    mpl.rcParams["xtick.major.size"] = 5
    mpl.rcParams["xtick.major.width"] = 2.5
    mpl.rcParams["ytick.major.size"] = 5
    mpl.rcParams["ytick.major.width"] = 2.5
    mpl.rcParams["axes.linewidth"] = 2.5
    plt.scatter(x, y, color="black")

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlim([0, 100])

    plt.tight_layout()
    plt.savefig(savename, dpi=300)

    plt.cla()
