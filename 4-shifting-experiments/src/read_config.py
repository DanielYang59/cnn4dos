#!/bin/usr/python3
# -*- coding: utf-8 -*-


import yaml
from pathlib import Path


def generate_config_template():
    pass


def read_config(filename: str) -> dict:
    """
    Read a YAML configuration file and return its content as a dictionary.

    Args:
        filename (str): The path to the YAML configuration file to read.

    Returns:
        dict: A dictionary containing the YAML configuration file's contents.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If the file is not a valid YAML file.
    """
    filepath = Path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"The file {filename} does not exist.")

    try:
        with filepath.open("r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")

    return config
