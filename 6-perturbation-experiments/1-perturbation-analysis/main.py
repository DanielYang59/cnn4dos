#!/usr/bin/env python3
# -*- coding: utf-8 -*-


dos_energy_range = (-14, 6, 4000)
plot_energy_range = (-5, 5)
project_data_dir = "data"
project_results_dir = "results"
extract_dos_from_vasprun_script = "../../0-utils/extract_dos_from_vasprunxml.py"
extract_single_atom_dos_script = "../../0-utils/extract_single_atom_DOS.py"


from pathlib import Path
from lib.analyze_dos_perturb import Analyzer
from lib.list_projects import list_projects
from lib.plot_dos_change import dosChangePlotter


# Main
if __name__ == "__main__":
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
