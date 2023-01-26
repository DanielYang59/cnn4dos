#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Configs
## Dataset loading
feature_dir = "../../0-dataset/feature_DOS"
label_dir = "../../0-dataset/label_adsorption_energy"


## Substrate and adsorbate selection
substrates = ["g-C3N4", "nitrogen-graphene", "vacant-graphene"]  # , "C2N", "BN", "BP"
substrates.extend(["aug-g-C3N4", "aug-nitrogen-graphene", "aug-vacant-graphene"])  # data augmentation
adsorbates = ["1-CO2", "2-COOH", "3-CO", "4-OCH", "11-HER"] # "5-OCH2", "6-OCH3", "7-O", "8-OH",


## Model training configs
batch_size = 16
validation_ratio = 0.2
append_adsorbate_dos = True
checkpoint_path = "checkpoint"


# Import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras_tuner
import tensorflow as tf
import numpy as np

from lib.dataset import Dataset
from lib.hp_model import hp_model


# Main Loop
if __name__ == "__main__":

    # Set global random seed
    tf.random.set_seed(0)
    np.random.seed(0)
    
    # Load dataset
    dataFetcher = Dataset()

    ## Load feature
    dataFetcher.load_feature(feature_dir, substrates, adsorbates, dos_filename="dos_up.npy", 
                             states={"is"}, # initial state only predictive model
                             )  
    print(f"A total of {dataFetcher.numFeature} samples loaded.")
    
    # # Normalize DOS
    # dataFetcher.scale_feature(mode="normalization") 
    
    
    ## append molecule DOS
    if append_adsorbate_dos:
        dataFetcher.append_adsorbate_DOS(adsorbate_dos_dir=os.path.join(feature_dir, "adsorbate-DOS"))  
    
    
    ## Load label
    dataFetcher.load_label(label_dir)

    ## Combine feature and label
    feature = dataFetcher.feature.values()
    label = dataFetcher.label.values()

    dataset = tf.data.Dataset.from_tensor_slices((feature, label))
    dataset = dataset.batch(batch_size)

    ## Train-validation split
    train_set, val_set = tf.keras.utils.split_dataset(dataset, right_size=validation_ratio, shuffle=True, seed=0)


    tuner = keras_tuner.RandomSearch(
    hypermodel=hp_model,
    objective="val_mean_absolute_error",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="hp_search",
    project_name="keras-tuner",
    )
    print(tuner.search_space_summary())
    
    
    tuner.search(train_set, validation_data=val_set, epochs=300, 
                 verbose=2,
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25),
                            
                            ],
                 )
    
    print(tuner.results_summary())
    