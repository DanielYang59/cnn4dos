#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import inquirer
import os


def list_projects(working_dir):
    # Read available projects from "data" dict
    subfolders = [f for f in os.listdir(working_dir) if os.path.isdir(os.path.join(working_dir, f)) and not f.startswith(".")]
    if subfolders:
        questions = [
                inquirer.List("project",
                message="Which project to analyze?",
                choices=subfolders,
                carousel=True),
                ]
        answers = inquirer.prompt(questions)
        
        return answers["project"]

    else:
        raise RuntimeError(f"No legal folder found under \"{working_dir}\".")