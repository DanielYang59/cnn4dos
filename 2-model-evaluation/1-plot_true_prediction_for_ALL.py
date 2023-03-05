#!/usr/bin/env python3
# -*- coding: utf-8 -*-


model_dir = "../1-model-and-training/2-best-model"


import os, sys
import yaml
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import r2_score
from matplotlib.patches import PathPatch

sys.path.append(model_dir)
from lib.dataset import Dataset


if __name__ == "__main__":
    # Load configs
    with open("config.yaml") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    ## paths
    feature_dir = cfg["path"]["feature_dir"]
    label_dir = cfg["path"]["label_dir"]
    ## species 
    substrates = cfg["species"]["substrates"]
    adsorbates = cfg["species"]["adsorbates"]
    centre_atoms = cfg["species"]["centre_atoms"]
    append_adsorbate_dos = cfg["species"]["append_adsorbate_dos"]
    load_augmentation = cfg["species"]["load_augmentation"]
    augmentations = cfg["species"]["augmentations"]
    spin = cfg["species"]["spin"]
    ## model training
    preprocessing = cfg["model_training"]["preprocessing"]
    remove_ghost = cfg["model_training"]["remove_ghost"] 

    
    # Import model
    model = tf.keras.models.load_model(os.path.join(model_dir, "model"))
    
    # Load dataset
    dataFetcher = Dataset()
    
    ## Load feature
    dataFetcher.load_feature(feature_dir, substrates, adsorbates, centre_atoms,
                            states={"is", }, spin=spin,
                            remove_ghost=remove_ghost, 
                            load_augment=load_augmentation, augmentations=augmentations)
    
    ## Append molecule DOS
    if append_adsorbate_dos:
        dataFetcher.append_adsorbate_DOS(adsorbate_dos_dir=os.path.join(feature_dir, "adsorbate-DOS"))
    
    ## Preprocess feature
    dataFetcher.scale_feature(mode=preprocessing)
    
    
    ## Load label
    dataFetcher.load_label(label_dir)

    ## Convert feature and label to array
    features = np.array(list(dataFetcher.feature.values()))
    labels = np.array(list(dataFetcher.label.values()))
    
    # Make predictions with model
    predictions = model.predict(features, verbose=0).flatten()
    

    # Change fonts to Helvetica 
    font = {'family' : 'sans-serif',
        'sans-serif':'Helvetica',
        'weight' : 'normal',
        'size'   : 18}
    plt.rc('font', **font)

     
    # Add grid for plotting
    g = sns.JointGrid(x=labels, y=predictions, ratio=4, 
                       height=6, 
                       space=0.2,
                       dropna=True,
                       )
    ## Set x/y axis range
    g.ax_joint.set_xlim([-11, 0.5])
    g.ax_joint.set_ylim([-11, 0.5])
    
    ## Set background color
    g.ax_joint.set_facecolor("#DAD9EB")  # background color
    
    
    # Add y=x line
    g.ax_joint.axline((0, 0), slope=1, zorder=1, color="#E69A8DFF", linewidth=8)
    
    # Plot scatter
    sns.scatterplot(x=labels, y=predictions, ax=g.ax_joint, s=200, alpha=0.5, color="#5F4B8BFF", zorder=2) # Ref: https://www.youtube.com/watch?v=t3G078DWXBM
    
    
    # Plot x-axis KDE distribution plot (top)
    ## Calculate estimation of distribution density
    kde_top_x = sm.nonparametric.KDEUnivariate(labels)
    kde_top_x.fit()  # Estimate the densities
    top_x = kde_top_x.support[kde_top_x.support <= 0]
    top_y = kde_top_x.density[:top_x.shape[0]]
    
    ## Create line
    g.ax_marg_x.plot(top_x, top_y, lw=0.2, color="blue")

    ## Create color filling
    top_img = g.ax_marg_x.imshow(top_y.reshape(1, -1), 
                    extent=[top_x[0], top_x[-1], 0, top_y.max()], 
                    cmap=plt.get_cmap('Blues'), alpha=0.9, 
                    aspect='auto', 
                    )
    top_img.set_clim(-0.15, 0.3)  # limit cmap range
    
    top_poly = g.ax_marg_x.fill_between(top_x, 0, top_y, color='none')
    top_img.set_clip_path(PathPatch(top_poly.get_paths()[0], transform=g.ax_marg_x.transData))
    g.ax_marg_x.set_ylim(0.0, 0.25)  # increase height

    # Plot x-axis KDE distribution plot (right)
    ## Calculate estimation of distribution density
    kde_right_y = sm.nonparametric.KDEUnivariate(predictions)
    kde_right_y.fit()  # Estimate the densities
    right_y = kde_right_y.support[kde_right_y.support <= 0]
    right_x = kde_right_y.density[:right_y.shape[0]]
    
    ## Create line
    g.ax_marg_y.plot(right_x, right_y, lw=0.2, color="red")

    ## Create color filling
    right_img = g.ax_marg_y.imshow(right_x.reshape(-1, 1)[::-1, :],
                    extent=[0, right_x.max(), right_y[0], 0], 
                    cmap=plt.get_cmap('Reds'), alpha=0.8,
                   aspect='auto',
                    )
    right_img.set_clim(-0.05, 0.3)

    
    right_poly = g.ax_marg_y.fill_betweenx(right_y, 0, right_x, color='none')  # check xy
    right_img.set_clip_path(PathPatch(right_poly.get_paths()[0], transform=g.ax_marg_y.transData))
    g.ax_marg_y.set_xlim(0.0, 0.25)  # increase width

    
    # Set x/y major ticks and font size
    g.ax_joint.set_xticks([-10, -5, 0])
    g.ax_joint.set_yticks([-10, -5, 0])
    g.ax_joint.tick_params(axis="both", labelsize=25, width=2.5, size=5)
    
    
    # Hide side plot ticks
    g.ax_marg_x.xaxis.set_ticks_position("none") 
    g.ax_marg_y.yaxis.set_ticks_position("none")
    
    # Add MAE text
    mae = np.absolute(np.subtract(labels, predictions)).mean()
    g.ax_joint.text(-8.5, -10, f"MAE = {'%.2f' % mae} eV", fontsize=28)
    
    # Calculate R2 score and print
    r2 = r2_score(y_true=labels, y_pred=predictions)
    print(f"R2 score is {r2}.")

    plt.tight_layout()
    plt.savefig(os.path.join("figures", "true_prediction_plot.png"), dpi=300)
    plt.show()
    