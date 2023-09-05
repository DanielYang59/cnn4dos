#!/bin/usr/python3
# -*- coding: utf-8 -*-


import keras
from pathlib import Path


def load_cnn_model(model_dir):
    """
    Load a Tensorflow/Keras CNN model from a local directory.

    Parameters:
        model_dir (str): The directory path where the TensorFlow/Keras model is stored.

    Returns:
        tensorflow.python.keras.engine.functional.Functional: The loaded CNN model.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        NotADirectoryError: If the specified path is not a directory.
    """
    # Create a Path object and check if it exists and is a directory
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"The specified directory {model_dir} does not exist.")

    if not model_path.is_dir():
        raise NotADirectoryError(f"The specified path {model_dir} is not a directory.")

    # Load the model using Keras
    cnn_model = keras.models.load_model("model")

    return cnn_model
