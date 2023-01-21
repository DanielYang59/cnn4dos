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
import tensorflow as tf
import numpy as np

from lib.dataset import Dataset
from lib.model import cnn_for_dos
from lib.record_runtime_info import record_system_info, record_python_package_ver


# Main Loop
if __name__ == "__main__":
    # # Save Python package version
    # record_python_package_ver()
    # # Save system information
    # record_system_info(logfile="sys_info.log")
    
    
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
    
    # Normalize DOS
    dataFetcher.scale_feature(mode="normalization") #DEBUG
    
    
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


    # Load model
    model = cnn_for_dos(
        input_shape=(4000, 9, 6),  # (NEDOS, numOrbital, numChannels)
        drop_out_rate=0.20,
        )
    

    # Compile model
    model.compile(
        # optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.mean_absolute_error, ],
    )

    # Callbacks
    ## Save best model
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True)
    ## Early Stop
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30)
    ## Learning Rate Schedule
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=10, factor=0.5, min_lr=1e-8)

    # Training model
    model.fit(train_set, validation_data=val_set, 
              verbose=2, 
              epochs=1000, 
              callbacks=[reduce_lr_callback,
                         early_stop_callback, 
                         # model_checkpoint_callback,
                         ],
              )
