# Occlusion Experiment for DOS Prediction

## Table of Contents

- [Overview](#overview)
- [Usage](#usage)
- [Structure](#structure)

## Overview

This experiment aims to explore the impact of occlusion on Density of States (DOS) predictions using Convolutional Neural Networks (CNN). We generate various occluded versions of the original DOS arrays and run predictions on each to assess the robustness and localization ability of the model.

## Usage

To run the experiment, execute the following command under a directory where a spin-up DOS is stored as dos_up.npy:

```bash
python main.py --config config.yaml
```

This will read the configuration from `config.yaml`, generate the occluded DOS arrays, and perform predictions.

## Structure

* `main.py`: The entry point of the experiment.
* `config.yaml`: The configuration file containing parameters for the experiment.
* `occlusionGenerator.py`: Contains the `occlusionGenerator` class for generating occluded DOS arrays.
* `CNNPredictor.py`: Contains the `CNNPredictor` class for making predictions using the loaded CNN model.
