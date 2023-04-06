
# Doping configs
atoms_to_replace = "GROUP"  # index starts from 1 ("GROUP" for automatic grouping)
new_element = "N"
source_dir = "0-原始模型-Cr-vac-graphene"
output_dir = "grouped_nitrogen-doped"


import os
import copy
import subprocess
from lib.poscar import POSCAR


if __name__ == "__main__":
    # Run group script to generate grouping of atoms
    if atoms_to_replace == "GROUP":
        atoms_to_replace = []
        grouping = subprocess.getstatusoutput("python3 group_atoms.py")
        if grouping[0] == 0:
            # Parse grouping results
            for line in grouping[1].split("\n")[1:]:
                atoms_to_replace.append(int(line.split(":")[1]))
    
    # Import source POSCAR
    assert os.path.exists(os.path.join(source_dir, "POSCAR"))
    source_poscar = POSCAR()
    source_poscar.read(os.path.join(source_dir, "POSCAR"))


    # # Loop through atoms to replace list
    for i in atoms_to_replace:
        # Check atom index
        assert isinstance(i, int) and i in range(1, sum(source_poscar.ion_numbers) + 1)
        
        doped_poscar = copy.copy(source_poscar)
        
        # Replace atom
        doped_poscar.replace_atom([i], new_element)
        
        # Write new POSCAR
        os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)
        doped_poscar.write(filename=os.path.join(output_dir, str(i), "POSCAR"))
