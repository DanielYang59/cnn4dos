#!/bin/usr/python3
# -*- coding: utf-8 -*-


from pathlib import Path


def get_folders_in_dir(directory_path: str, filter_file: str = None) -> list:
    """
    Get and return a list of folder names in the specified directory that optionally contain a specific file.

    Parameters:
        directory_path (str): The absolute or relative path to the directory in which to search for folders.
        filter_file (str, optional): A filename to filter the folders by. If specified, only folders containing
                                     this file will be returned. Default is None, meaning all folders will be returned.

    Returns:
        list: A list of folder paths that match the criteria. Each folder path is a `Path` object.

    Raises:
        FileNotFoundError: If no suitable folders are found, especially when a filter_file is specified and not found.
    """

    folder_list = []
    dir_path = Path(directory_path)

    # Check if the directory exists
    if not dir_path.exists():
        return f"Directory {directory_path} does not exist."


    # List items in the directory
    for item in dir_path.iterdir():
        if item.is_dir():
            # If a filter_file is specified, check if the folder contains it
            if filter_file:
                filter_file_path = item / filter_file
                if filter_file_path.exists():
                    folder_list.append(dir_path / item.name)
            else:
                folder_list.append(dir_path / item.name)

    if not folder_list:
        raise FileNotFoundError(f"No suitable folders containing {filter_file} found in ({directory_path}).")

    return folder_list
