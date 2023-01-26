#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import yaml

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
    
    
    # Load configs
    with open("config.yaml") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    ## paths
    feature_dir = cfg["path"]["feature_dir"]
    label_dir = cfg["path"]["label_dir"]
    ## species 
    substrates = cfg["species"]["substrates"]
    adsorbates = cfg["species"]["adsorbates"]
    append_adsorbate_dos = cfg["species"]["append_adsorbate_dos"]
    load_augmentation = cfg["species"]["load_augmentation"]
    augmentations = cfg["species"]["augmentations"]
    spin = cfg["species"]["spin"] 
    ## model training
    preprocessing = cfg["model_training"]["preprocessing"]
    batch_size = cfg["model_training"]["batch_size"]
    validation_ratio = cfg["model_training"]["validation_ratio"]
    epochs = cfg["model_training"]["epochs"]
    
     
    # Load dataset
    dataFetcher = Dataset()

    ## Load feature
    dataFetcher.load_feature(feature_dir, substrates, adsorbates, 
                             states={"is", }, spin=spin,
                             load_augment=load_augmentation, augmentations=augmentations)  
    print(f"A total of {dataFetcher.numFeature} samples loaded.")
    
    # Normalize DOS
    dataFetcher.scale_feature(mode=preprocessing)
    
    
    # Append molecule DOS
    if append_adsorbate_dos:
        dataFetcher.append_adsorbate_DOS(adsorbate_dos_dir=os.path.join(feature_dir, "adsorbate-DOS"))  
    
    
    ## Load label
    dataFetcher.load_label(label_dir)

    ## Combine feature and label
    feature = np.array(list(dataFetcher.feature.values()))
    label = np.array(list(dataFetcher.label.values()))
    ### Check input shape from data and from user input
    input_shape = feature[0].shape
    print(f"Found input data in {input_shape} shape, (NEDOS, numOrbital, numChannels).")

    dataset = tf.data.Dataset.from_tensor_slices((feature, label))
    dataset = dataset.batch(batch_size)

    ## Train-validation split
    train_set, val_set = tf.keras.utils.split_dataset(dataset, right_size=validation_ratio, shuffle=True)


    # Load model
    model = cnn_for_dos(
        input_shape=input_shape,  # (NEDOS, numOrbital, numChannels)
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
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoint", monitor="val_loss", save_best_only=True)
    ## Early Stop
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30)
    ## Learning Rate Schedule
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=10, factor=0.5, min_lr=1e-8)

    # Training model
    model.fit(train_set, validation_data=val_set, epochs=epochs,
              verbose=2, 
              callbacks=[reduce_lr_callback,
                         early_stop_callback, 
                         # model_checkpoint_callback,
                         ],
              )
