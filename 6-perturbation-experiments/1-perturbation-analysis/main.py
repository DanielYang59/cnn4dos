#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import yaml
from lib.analyze_dos_perturb import Analyzer
from lib.list_projects import list_projects
from lib.plot_dos_change import dosChangePlotter


# Main
if __name__ == "__main__":
    # Load configs
    with open("config.yaml") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    project_data_dir = Path(cfg["path"]["project_data_dir"])
    project_results_dir = Path(cfg["path"]["project_results_dir"])
    extract_dos_from_vasprun_script = Path(cfg["path"]["extract_dos_from_vasprun_script"])
    extract_single_atom_dos_script = Path(cfg["path"]["extract_single_atom_dos_script"])

    dos_energy_range = cfg["ranges"]["dos_energy_range"]
    plot_energy_range = cfg["ranges"]["plot_energy_range"]


    # Let user choose project to work on
    project = list_projects(Path(project_data_dir))


    # Work on selected project
    atom_index = int(input("Extract DOS of which atom? (starts from 1)"))
    analyzer = Analyzer(rootpath=Path(project_data_dir) / project,
                        atom_index=atom_index
                        )


    # Extract DOS of selected atom
    analyzer.extract_dos(Path(extract_dos_from_vasprun_script).resolve(),
                         Path(extract_single_atom_dos_script).resolve(),
                         verbose=True
                         )


    # Calculate DOS change
    analyzer.calculate_dos_change(energy_range=dos_energy_range,
                                  save_interpolated=True
                                  )


    # Plot DOS change at selected energy range
    plotter = dosChangePlotter(
        project_dir=Path(project_results_dir) / project,
        energy_range=plot_energy_range
        )
