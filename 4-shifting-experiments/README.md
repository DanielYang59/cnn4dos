
# Shifting Experiment

## Overview

This project aims to perform a shifting experiment involving Density of States (DOS) and adsorbate DOS arrays.

## Features

1. Load and preprocess DOS and adsorbate DOS arrays.
2. Generate shifted DOS arrays based on user-defined criteria.
3. Use a Convolutional Neural Network (CNN) model for predictions.
4. Dynamically generate visualizations of the shifting experiment results using `ShiftPlotter`.

## How to Run

1. Ensure you have all required packages installed.
2. Place your configuration in `config.yaml`.
3. Run `main.py`.

## Classes and Functions

- `DataLoader`: Load and preprocess DOS and adsorbate DOS arrays.
- `DosProcessor`: Load DOS arrays from specified folders.
- `ShiftGenerator`: Generate shifted DOS arrays.
- `CNNPredictor`: Use a pre-trained CNN model for predictions.
- `ShiftPlotter`: Visualize the shifting experiments. Allows for flexible configurations including changing colormap and saving to different directories.
- `get_folders_in_dir`: Utility function to get folders matching certain criteria.

## Configuration

A sample configuration file `config.yaml` is included. It specifies the working directory, CNN model path, shifting parameters, and other options for the shifting experiment.

### ShiftPlotter Configuration

To use `ShiftPlotter`, you can optionally specify the following:

- `colormap`: Changes the color map used for the visualization. Default is "magma".
- `figures`: Changes the directory where the generated figures will be saved. Default is the current directory.

To specify these options, include them when initializing `ShiftPlotter`.

Example:

```python
plotter = ShiftPlotter(all_predictions, config)
plotter.plot(colormap="viridis", figures=Path("path/to/figures"))
```
