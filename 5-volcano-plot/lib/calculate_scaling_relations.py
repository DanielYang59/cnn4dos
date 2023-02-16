

import csv
import os


def calculate_scaling_relations(free_energy_linear_relation, molecule_energy_file, reaction_pathway_file):
    # Check args
    assert isinstance(free_energy_linear_relation, dict)
    assert os.path.exists(molecule_energy_file)
    assert os.path.exists(reaction_pathway_file)

    
    # Import molecule energy
    molecule_energy_dict = import_molecule_energy(molecule_energy_file)
    
    
    # Import reaction path
    
    
    # 


def import_molecule_energy(file):
    # Import molecule energy csv file
    molecule_energy_dict = {}
    with open(file) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for line in reader:
            molecule_energy_dict[line[0]] = float(line[1])
    
    return molecule_energy_dict
        

# Test area
if __name__ == "__main__":
    calculate_scaling_relations(
        {}, #DEBUG 
        "../data/energy_molecule.csv",
        "../data/reaction_pathway.json",
    )
