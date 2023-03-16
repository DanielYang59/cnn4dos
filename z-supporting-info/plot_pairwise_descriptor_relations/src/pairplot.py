#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
    
    
class PairPlot:
    def __init__(self, data) -> None:
        assert isinstance(data, pd.DataFrame)
        self.data = data
        
        
    def __preprocess_data(self, descriptors):
        # Drop metal/substrate name columns for plotting
        self.data.drop(labels=["metal", "substrate"], axis=1, inplace=True)
        
        
        # Drop non-numerical columns
        self.data = self.data._get_numeric_data()
        
        
        # Drop lines with missing value
        self.data.dropna(axis=0, inplace=True)
        
        
        # Take selected descriptors only
        self.data = self.data.loc[:, descriptors]
    
    
    def plot_pairplot(self, descriptors, savename, show=False):
        """Generate pairwise relations plot.

        Args:
            descriptors (_type_): _description_
            
        Notes:
            Ref: https://seaborn.pydata.org/generated/seaborn.pairplot.html
            
        """
        # Preprocess datasheet
        self.__preprocess_data(descriptors=descriptors)
  
        
        # Generate pairwise relation plot
        
        #sns.set(style="ticks", color_codes=True)

        sns.pairplot(data=self.data,
            corner=False,  # show lower triangle only
            )
        
        
        # Save and show figure
        plt.savefig(savename, dpi=300)
        if show:
            plt.show()
