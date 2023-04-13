# Configs
source_substrates_dir = "test_substrate"
metal_atom_index = 71  # starts from 1

target_structure_dir = "target_dir"
is_distance = 6
fs_distance = 2.5
poscar_lib = "poscar_lib"  # POSCAR is named as 1-CO2_96-101-102, where 96/101/102 are adsorbate atoms and 96 is reference atom
adsorbates = ["3-CO", ]
substrate_poscar_name = "POSCAR_substrate"


adsorbkit_path = "lib/adsorbkit.py"


import os
import shutil
import time
from lib.adsorbin_template import adsorbin_template
from lib.poscar import POSCAR


def create_structures(substrates_dir, adsorbate_poscar, target_dir, substrate_poscar_filename, adsorbin_para, put_substrate_to_centre=True):
    # Check args
    assert os.path.isdir(substrates_dir)
    assert os.path.exists(adsorbate_poscar)
    os.makedirs(target_dir, exist_ok=True)

    # Loop through substrates
    for sub in os.listdir(substrates_dir):
        if os.path.exists(os.path.join(substrates_dir, sub, substrate_poscar_filename)):
            # Create dir for substrate
            os.makedirs(os.path.join(target_dir, sub), exist_ok=True)

            # Copy substrate POSCAR
            shutil.copy(os.path.join(substrates_dir, sub, substrate_poscar_filename), os.path.join(target_dir, sub, substrate_poscar_filename))

            # Copy adsorbate POSCAR
            shutil.copy(adsorbate_poscar, os.path.join(target_dir, sub, "POSCAR_adsorbate"))

            # Get AdsorbIN template
            adsorbin = adsorbin_template.split("\n")

            # Modify AdsorbIN template
            adsorbin[3] = "adsorb_style = top"
            adsorbin[4] = f"adsorb_distance = {adsorbin_para['distance']}"
            adsorbin[6] = f"substrate = {substrate_poscar_filename}"
            adsorbin[7] = f"substrate_ref = {adsorbin_para['substrate_ref']}"
            adsorbin[8] = "adsorbate = POSCAR_adsorbate"

            ## Get adsorbate atoms from POSCAR filename
            adsorbate_atoms = adsorbate_poscar.split(os.sep)[-1].split("_")[-1].split("-")
            adsorbin[9] = f"adsorbate_select = {','.join(adsorbate_atoms)}"
            adsorbin[10] = f"adsorbate_ref = {adsorbate_atoms[0]}"

            # Write AdsorbIN to file
            with open(os.path.join(target_dir, sub, "AdsorbIN"), mode="w") as f:
                f.write("\n".join(adsorbin))


            # Change to adsorbkit working dir
            previous_path = os.getcwd()
            os.chdir(os.path.join(target_dir, sub))

            # Put substrate to bottom, add adsorbate and then put everything to centre
            if put_substrate_to_centre:
                # Run poscarkit to put substrate to bottom
                poscar = POSCAR()
                poscar.read(os.path.join(os.getcwd(), substrate_poscar_filename))
                poscar.reposition_all(mode="bottom", distance=None)
                poscar.write(filename=substrate_poscar_filename)

                # Run adsorbkit
                os.system(f"python3 {adsorbkit_path}")

                # Run poscarkit to put everything at centre
                poscar = POSCAR()
                poscar.read(os.path.join(os.getcwd(), "POSCAR"))
                poscar.reposition_all(mode="centre", distance=None)
                poscar.write(filename="POSCAR")


            else:
                os.system(f"python3 {adsorbkit_path}")


            # Change back to previous dir
            os.chdir(previous_path)


if __name__ == "__main__":
    # Check configs
    assert os.path.isdir(source_substrates_dir)
    assert os.path.isdir(target_structure_dir)
    assert os.path.isdir(poscar_lib)

    assert isinstance(metal_atom_index, int)
    assert isinstance(is_distance, (float, int))
    assert isinstance(fs_distance, (float, int))


    # Verbose
    print(f"Metal atom index is {metal_atom_index} (starts from 1). Starting in 3 seconds.")
    time.sleep(3)

    # Loop through list of adsorbates
    for ads in adsorbates:
        # Match adsorbate POSCAR file name
        for file in os.listdir(poscar_lib):
            if file.startswith(ads):
                ads_file = file


        # Generate initial state structures
        adsorbin_para_is = {"distance":str(is_distance), "substrate":substrate_poscar_name, "substrate_ref":str(metal_atom_index), "adsorbate":"POSCAR_adsorbate"}

        create_structures(substrates_dir=source_substrates_dir, adsorbate_poscar=os.path.join(poscar_lib, ads_file), target_dir=os.path.join(target_structure_dir, source_substrates_dir.split(os.sep)[-1], "1-is", ads_file), substrate_poscar_filename=substrate_poscar_name, adsorbin_para=adsorbin_para_is)


        # Generate final state structures
        adsorbin_para_fs = {"distance":str(fs_distance), "substrate":substrate_poscar_name, "substrate_ref":str(metal_atom_index), "adsorbate":"POSCAR_adsorbate"}

        create_structures(substrates_dir=source_substrates_dir, adsorbate_poscar=os.path.join(poscar_lib, ads_file), target_dir=os.path.join(target_structure_dir, source_substrates_dir.split(os.sep)[-1], "2-fs", ads_file), substrate_poscar_filename=substrate_poscar_name, adsorbin_para=adsorbin_para_fs)
