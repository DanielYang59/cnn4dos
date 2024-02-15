"""Utility script for getting the prediction metrics of CNN model."""


import os

import numpy as np

import sys
import warnings

import tensorflow as tf
import yaml
from lib.dataset import Dataset
from sklearn.metrics import r2_score
from tensorflow import keras


# Main Loop
if __name__ == "__main__":
    # Set global random seed
    tf.random.set_seed(0)
    np.random.seed(0)

    # Load configs
    with open("config.yaml", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    # paths
    feature_dir = cfg["path"]["feature_dir"]
    label_dir = cfg["path"]["label_dir"]
    # species
    substrates = cfg["species"]["substrates"]
    adsorbates = cfg["species"]["adsorbates"]
    centre_atoms = cfg["species"]["centre_atoms"]
    append_adsorbate_dos = cfg["species"]["append_adsorbate_dos"]
    load_augmentation = cfg["species"]["load_augmentation"]
    augmentations = cfg["species"]["augmentations"]
    spin = cfg["species"]["spin"]
    # model training
    preprocessing = cfg["model_training"]["preprocessing"]
    remove_ghost = cfg["model_training"]["remove_ghost"]
    batch_size = cfg["model_training"]["batch_size"]
    validation_ratio = cfg["model_training"]["validation_ratio"]
    epochs = cfg["model_training"]["epochs"]
    sample_size = cfg["model_training"]["sample_size"]

    # Load features(DOS) and labels from cached file to save time
    if os.path.exists("features.npy") and os.path.exists("labels.npy"):
        warnings.warn(
            "Warning! features/labels load from cached file. Tags changed after cache generation in config.yaml might not take effect."
        )
        features = tf.convert_to_tensor(np.load("features.npy"))
        labels = tf.convert_to_tensor(np.load("labels.npy"))

        total_sample = labels.shape[0]

    else:
        # Load dataset
        dataFetcher = Dataset()

        # Load feature
        dataFetcher.load_feature(
            feature_dir,
            substrates,
            adsorbates,
            centre_atoms,
            states={
                "is",
            },
            spin=spin,
            remove_ghost=remove_ghost,
            load_augment=load_augmentation,
            augmentations=augmentations,
        )
        if sample_size == "ALL":
            print(f"A total of {dataFetcher.numFeature} samples loaded.")
        elif isinstance(sample_size, int) and sample_size >= 1:
            print(
                f"A total of {dataFetcher.numFeature} samples found, {sample_size} loaded."
            )
        else:
            raise ValueError('sample_size should be "ALL" or an interger.')

        # Append molecule DOS
        if append_adsorbate_dos:
            dataFetcher.append_adsorbate_DOS(
                adsorbate_dos_dir=os.path.join(feature_dir, "adsorbate-DOS")
            )

        # Load label
        dataFetcher.load_label(label_dir)

        # Combine feature and label
        features = np.array(list(dataFetcher.feature.values()))
        labels = np.array(list(dataFetcher.label.values()))
        np.save("features.npy", features)
        np.save("labels.npy", labels)

        print("Cache generated. Exiting...")
        sys.exit()

    # Load best model
    model = keras.models.load_model("model")
    model.summary()

    # Predict with best model
    predictions = model.predict(features, verbose=0).flatten()

    # Evaluate prediction
    mae = np.absolute(np.subtract(labels, predictions)).mean()
    print(f"MAE is {mae} eV.")

    # Calculate R2 score
    r2 = r2_score(y_true=labels, y_pred=predictions)
    print(f"R2 score is {r2}.")
