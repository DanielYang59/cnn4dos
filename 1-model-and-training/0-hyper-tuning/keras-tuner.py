#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Import packages
import os
import numpy as np
import yaml
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import keras_tuner

from lib.dataset import Dataset
from lib.hp_model import hp_model


# Main Loop
if __name__ == "__main__":
    # DEBUG: check running device
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
    feature = dataFetcher.feature.values()
    label = dataFetcher.label.values()

    dataset = tf.data.Dataset.from_tensor_slices((feature, label))
    dataset = dataset.shuffle(reshuffle_each_iteration=False)
    
    ## Take a subset if required
    if sample_size != "ALL":
        dataset = dataset.take(sample_size)
    
    dataset = dataset.batch(batch_size)


    # Train-validation split
    numTrainBatches = int((1 - validation_ratio) * (dataFetcher.numFeature / batch_size))  # take by number of "batches", not samples
    train_set = dataset.take(numTrainBatches)
    val_set = dataset.skip(numTrainBatches)
    

    # # Legacy train-validation split # NOT available in TensorFlow 2.9
    # train_set, val_set = tf.keras.utils.split_dataset(dataset, right_size=validation_ratio, shuffle=True, seed=0)  
    

    # Hyper Tuning with Keras Tuner
    tuner = keras_tuner.Hyperband(
    hypermodel=hp_model,
    objective="val_mean_absolute_error",
    directory="hp_search",
    )
    print(tuner.search_space_summary())
    
    
    tuner.search(train_set, validation_data=val_set, epochs=300, 
                 verbose=2,
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25),
                            
                            ],
                 )
    
    print(tuner.results_summary())
    