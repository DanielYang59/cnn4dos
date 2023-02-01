#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras_tuner
import tensorflow as tf
import numpy as np
import yaml

from lib.dataset import Dataset
from lib.hp_model import hp_model


# Main Loop
if __name__ == "__main__":
    # Set global random seed
    tf.random.set_seed(0)
    np.random.seed(0)
    
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
    batch_size = cfg["model_training"]["batch_size"]
    validation_ratio = cfg["model_training"]["validation_ratio"]
    epochs = cfg["model_training"]["epochs"]
    
    
    # Load dataset
    dataFetcher = Dataset()

    ## Load feature
    dataFetcher.load_feature(feature_dir, substrates, adsorbates, centre_atoms,
                             states={"is", }, spin=spin,
                             remove_ghost=remove_ghost, 
                             load_augment=load_augmentation, augmentations=augmentations) 
    print(f"A total of {dataFetcher.numFeature} samples loaded.")
    
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

    # Train-validation split
    train_set, val_set = tf.keras.utils.split_dataset(dataset, right_size=validation_ratio, shuffle=True, seed=0)


    # Hyper Tuning with Keras Tuner
    tuner = keras_tuner.BayesianOptimization(
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
    