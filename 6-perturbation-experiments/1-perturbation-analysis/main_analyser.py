#!/usr/bin/env python3
# -*- coding: utf-8 -*-

project_dir = "data"



import os
from pathlib import Path


# Main
if __name__ == "__main__":
    # Read available projects from "data" dict
    subfolders = [f.path for f in os.scandir(project_dir) if f.is_dir() and not f.startswith(".")]
    print(subfolders)
    

