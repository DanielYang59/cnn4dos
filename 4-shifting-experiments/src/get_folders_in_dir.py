#!/bin/usr/python3
# -*- coding: utf-8 -*-


from pathlib import Path


def get_folders_in_dir(directory_path: str, filter_file: str = None) -> list:
    """
    Get and return a list of folder names in the specified directory.

    Args:
        directory_path (str): The path to the directory to look in.
        filter_file (str, optional): The name of a file to filter folders by.
                                     Only folders containing this file will be returned.
                                     Defaults to None.

    Returns:
        list: A list of folder names that match the criteria.
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
                    folder_list.append(item.name)
            else:
                folder_list.append(item.name)

    return folder_list
