#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Import packages
import os
import numpy as np
import yaml
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import keras_tuner

from lib.dataset import Dataset
from lib.model import cnn_for_dos


# Main Loop
if __name__ == "__main__":
    # # Print GPU info
    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # print("Device name: ", tf.test.gpu_device_name())
    
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
    sample_size = cfg["model_training"]["sample_size"]
    
    
    # Load dataset
    dataFetcher = Dataset()

    ## Load feature
    dataFetcher.load_feature(feature_dir, substrates, adsorbates, centre_atoms,
                             states={"is", }, spin=spin,
                             remove_ghost=remove_ghost, 
                             load_augment=load_augmentation, augmentations=augmentations)
    if sample_size == "ALL":
        print(f"A total of {dataFetcher.numFeature} samples loaded.")
    elif isinstance(sample_size, int) and sample_size >= 1:
        print(f"A total of {dataFetcher.numFeature} samples found, {sample_size} loaded.")
    else:
        raise ValueError('sample_size should be "ALL" or an interger.')
    
    ## Append molecule DOS
    if append_adsorbate_dos:
        dataFetcher.append_adsorbate_DOS(adsorbate_dos_dir=os.path.join(feature_dir, "adsorbate-DOS"))
    
    
    ## Load label
    dataFetcher.load_label(label_dir)

    ## Combine feature and label
    feature = np.array(list(dataFetcher.feature.values()))
    label = np.array(list(dataFetcher.label.values()))

    dataset = tf.data.Dataset.from_tensor_slices((feature, label))
    dataset = dataset.shuffle(buffer_size=dataFetcher.numFeature, reshuffle_each_iteration=False)
   
    ## Take a subset if required
    if sample_size != "ALL":
        dataset = dataset.take(sample_size)
    
    # Train-validation split
    train_size = int(dataFetcher.numFeature * (1 - validation_ratio))
    train_set = dataset.take(train_size)
    val_set = dataset.skip(train_size)
    
    # Batch and prefetch
    train_set = train_set.batch(batch_size=batch_size) 
    train_set = train_set.prefetch(tf.data.AUTOTUNE)
    
    val_set = val_set.batch(batch_size)
    val_set = val_set.prefetch(tf.data.AUTOTUNE)


    # Load model
    model = cnn_for_dos(
        input_shape=(4000, 9, 6),  # (NEDOS, numOrbital, numChannels)
        drop_out_rate=0.25,
        )
    
     # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0015),
        # optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.mean_absolute_error, ],
    )
    
    # Callbacks
    ## Save best model
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoint", monitor="val_loss", save_best_only=True)
    ## Early Stop
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50)
    ## Learning Rate Schedule
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=10, factor=0.5, min_lr=1e-6)

    # Training model
    model.fit(train_set, validation_data=val_set, epochs=epochs,
              verbose=2, 
              callbacks=[reduce_lr_callback,
                         early_stop_callback, 
                         # model_checkpoint_callback,
                         ],
              )
    
