#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import matplotlib.pyplot as plt
import seaborn as sns

from src.create_heatmap import corrplot
from src.generate_corr_matrix import generate_corr_matrix
from src.format_mol_name import format_mol_name


if __name__ == "__main__":
    # Generate correlation matrix
    corr_matrix = generate_corr_matrix(config_file="config.yaml")
    
    
    # Reformat molecule names
    revised_names = [format_mol_name(name) for name in corr_matrix.index.values]

    # Apply new names
    corr_matrix.index = revised_names
    corr_matrix.columns = revised_names
    
    
    # Plot correlation map
    sns.set(color_codes=True, font_scale=1.2)
    plt.figure(figsize=(8, 7))
    corrplot(corr_matrix,
            size_scale=500, marker="s",  # shape of the marker
             )
    
    plt.savefig(os.path.join("figures", "Eads_correlation_map.png"), bbox_inches="tight", dpi=300)
    
    
    # Show correlation coefficients
    print(corr_matrix)
    