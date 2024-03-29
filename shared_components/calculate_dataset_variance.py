#!/usr/bin/env python3
# -*- coding: utf-8 -*-


training_dir = "../1-model-and-training/1-hyper-tune"


import numpy as np
from pathlib import Path
import yaml

import sys
sys.path.append(training_dir)
from lib.dataset import Dataset


if __name__ == "__main__":
    # Load configs
    with open(Path(training_dir) / "config.yaml") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    ## paths
    feature_dir = Path("../dataset/feature_DOS")
    label_dir = Path("../dataset/label_adsorption_energy")
    ## species
    substrates = cfg["species"]["substrates"]
    adsorbates = cfg["species"]["adsorbates"]
    centre_atoms = cfg["species"]["centre_atoms"]
    load_augmentation = cfg["species"]["load_augmentation"]
    augmentations = cfg["species"]["augmentations"]
    spin = cfg["species"]["spin"]


    # Load dataset
    dataFetcher = Dataset()

    ## Load feature
    dataFetcher.load_feature(feature_dir,
                             substrates, adsorbates,
                             centre_atoms,
                             states={"is", },
                             spin=spin,
                             load_augment=load_augmentation, augmentations=augmentations)

    ## Load labels
    dataFetcher.load_label(label_dir)
    label = np.array(list(dataFetcher.label.values()))

    # Calculate and print variance
    print(f"A total of {dataFetcher.numFeature} samples loaded.")
    print(f"Standard deviation is {np.std(label)}, variance is {np.var(label)}.")
