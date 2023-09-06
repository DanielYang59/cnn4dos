# Shifting Experiment

## Overview

This project aims to perform a shifting experiment involving Density of States (DOS) and adsorbate DOS arrays.

## Features

1. Load and preprocess DOS and adsorbate DOS arrays.
2. Generate shifted DOS arrays based on user-defined criteria.
3. Use a Convolutional Neural Network (CNN) model for predictions.

## How to Run

1. Ensure you have all required packages installed.
2. Place your configuration in `config.yaml`.
3. Run `main.py`.

## Classes and Functions

- `DataLoader`: Load and preprocess DOS and adsorbate DOS arrays.
- `DosProcessor`: Load DOS arrays from specified folders.
- `ShiftGenerator`: Generate shifted DOS arrays.
- `CNNPredictor`: Use a pre-trained CNN model for predictions.
- `get_folders_in_dir`: Utility function to get folders matching certain criteria.

## Configuration

A sample configuration file `config.yaml` is included. It specifies the working directory, CNN model path, and other parameters for the shifting experiment.
