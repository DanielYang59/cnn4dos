#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_scatter(x, y, show=False, savename="d-band-Eads.png", dpi=300):
    # Check args
    assert len(x) == len(y)
    
    # Create scatter plot
    mpl.rcParams["axes.linewidth"] = 2.5
    plt.scatter(x, y, )
    
    
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
