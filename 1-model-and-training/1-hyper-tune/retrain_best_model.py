#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Import packages
import os, sys
import numpy as np
import yaml
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import keras_tuner
from datetime import datetime

from hp_model import hp_model
from lib.dataset import Dataset


# Main Loop
if __name__ == "__main__":
    # Print GPU info
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Device name: ", tf.test.gpu_device_name())
    
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
    
    
    # Load features(DOS) and labels from cached file to save time
    if os.path.exists("features.npy") and os.path.exists("labels.npy"):
        print("Warning! features/labels load from cached file. Tags changed after cache generation in config.yaml might not take effect.")
        feature = tf.convert_to_tensor(np.load("features.npy"))
        label = tf.convert_to_tensor(np.load("labels.npy"))
        
        total_sample = label.shape[0]
 
    
    else:
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
        np.save("features.npy", feature)
        np.save("labels.npy", label)
        
        print("Cache generated. Exiting...")
        sys.exit()
        
        total_sample = dataFetcher.numFeature


    dataset = tf.data.Dataset.from_tensor_slices((feature, label))
    dataset = dataset.shuffle(buffer_size=total_sample, reshuffle_each_iteration=False)
   
    ## Take a subset if required
    if sample_size != "ALL":
        dataset = dataset.take(sample_size)
    
    # Train-validation split
    train_size = int(total_sample * (1 - validation_ratio))
    train_set = dataset.take(train_size)
    val_set = dataset.skip(train_size)
    
    # Batch and prefetch
    train_set = train_set.batch(batch_size=batch_size)
    train_set = train_set.prefetch(tf.data.AUTOTUNE)
    
    val_set = val_set.batch(batch_size)
    val_set = val_set.prefetch(tf.data.AUTOTUNE)


    # Initiate HyperBand Tuner
    tuner = keras_tuner.Hyperband(
        hypermodel=hp_model,
        max_epochs=200,
        factor=3,
        overwrite=False,
        objective="val_mean_absolute_error",
        directory="hp_search",
        project_name="best_model",
        )
    
    print("search space: ", tuner.search_space_summary())
    
    
    # Retrain the best model
    # Ref: https://keras.io/guides/keras_tuner/getting_started/
    best_hps = tuner.get_best_hyperparameters(0)
    model = hp_model(best_hps)
    
    
    # Callbacks
    ## Save best model
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoint", monitor="val_loss", save_best_only=True)
    ## Early Stop
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50)
    ## Learning Rate Schedule
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=10, factor=0.5, min_lr=1e-6)
    ## Add TensorBoard Callback
    ## Ref:https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras
    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tblogs",
                                                    histogram_freq=1,
                                                    profile_batch=(1, 10))
    
    
    # Start model training
    model.fit(train_set, validation_data=val_set, epochs=epochs,
              verbose=2, 
              callbacks=[reduce_lr_callback,
                         # tboard_callback, 
                         early_stop_callback, 
                         # model_checkpoint_callback,
                         ],)
