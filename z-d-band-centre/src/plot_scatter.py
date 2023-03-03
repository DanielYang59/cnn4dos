#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_scatter(x, y, labels, colors=None, show=False, savename="d-band-Eads.png", dpi=300):
    """Plot scatter for d-band centre vs adsorption energy relationship.

    Args:
        x (list): x coordinates
        y (list): y coordinates
        labels (list): 
        colors ((list, None), optional): list of colors for each point or None for default color. Defaults to None.
        show (bool, optional): show plot. Defaults to False.
        savename (str, optional): figure saving name. Defaults to "d-band-Eads.png".
        dpi (int, optional): dpi. Defaults to 300.
        
    """
    # Check args
    assert len(x) == len(y)
    
    if colors is None:
        colors = "black"
    else:
        assert len(x) == len(colors)
    
    
    # Create scatter plot
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams["axes.linewidth"] = 2.5
    plt.scatter(x, y, c="darkblue")
    # for i in range(len(x)):
    #     plt.scatter(x[i], y[i], c=colors[i], label=labels[i])
    
    # Set legend
    plt.legend()
    
    
    # Set x/y axis labels
    plt.xlabel("d-Band Centre (eV)", fontsize=18)
    plt.ylabel("Adsorption Energy (eV)", fontsize=16)
    
    # Set x/y tick label sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Set x/y tick length and thickness
    plt.tick_params("both", length=5, width=2.5, which="major")
    
    
    # Show and save figure
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(savename, dpi=dpi)
    plt.cla()
