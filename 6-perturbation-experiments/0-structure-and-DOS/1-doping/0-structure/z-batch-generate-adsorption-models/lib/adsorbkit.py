#!/usr/bin/env python3
""" Brief Intro:
Add adsorbate to substrate, by YangHy.
If you find this script helpful or have any issues/advice to report,
please kindly reach out to me at yanghaoyu97@outlook.com.
"""

auto_poscar_rearrange = False  # turn on/off auto POSCAR rearrange after POSCAR combination (recommended off)


import os
import sys
import time
import numpy as np
import copy
import math


# Calculate distance between 2 atoms
def calc_distance(coordinate1, coordinate2):
    distance = math.sqrt((coordinate1[0] - coordinate2[0]) ** 2 + (coordinate1[1] - coordinate2[1]) ** 2 + (coordinate1[2] - coordinate2[2]) ** 2)
    return distance


def translate_atoms(atom_select, atom_list, index_start=0):
    """
    Translate "atoms" into "atom_index"
    :param atom_select:
    :param atom_list:
    :param index_start: where the output index list to start
    :return:
    """
    dosin_atom = atom_select.lower().replace(" ", "").split(",")
    atom_index = []
    for index in dosin_atom:
        if index[0].isalpha():
            if index == "all":
                atom_index = list(range(len(atom_list)))
                break
            else:
                for atom_list_i in list(range(len(atom_list))):
                    if atom_list[atom_list_i].lower() == index:
                        atom_index.append(atom_list_i)
        elif index[0].isdigit():  # Caution! Atom list starting from "0"!
            if "-" in index:  # if comes in the "range" form ("1-20")
                atom_index.extend(list(range(int(index.split("-")[0]) - 1, int(index.split("-")[1]))))
            else:
                atom_index.append(int(index) - 1)
        else:
            print("Please check your atom assignment in AdsorbIN!")
            sys.exit(0)

    # Check repeated elements in the atom index
    atom_index = [int(i) for i in atom_index]
    if not len(atom_index) == len(set(atom_index)):
        print("Reduplicative atom found in AdsorbIN! Please check!")
        sys.exit(0)

    # Adjust atom index starting value
    atom_index_output = [(i + index_start) for i in atom_index]
    return atom_index_output  # indexes starting from "0" at default!


def read_adsorbin():
    """
    Read control parameters from "AdsorbIN"
    :return: a dictionary containing all parameters in "AdsorbIN"
    """
    if os.path.exists("AdsorbIN"):
        with open("AdsorbIN", mode="r", encoding="utf-8") as adinf:
            # Sort parameters in AdsorbIN into dict
            adsorb_in = {}
            for line in [line.strip() for line in adinf]:
                if not (line.startswith("#") or (len(line) == 0)):
                    adsorb_in[line.replace(" ", "").split("=")[0]] = line.replace(" ", "").split("=")[1].split("#")[0]  # remove the comment at the end of the line

            # Check for empty values in AdsorbIN
            for key in adsorb_in:
                if len(adsorb_in[key]) == 0:
                    print(f"Warning! Empty value detected at {key} in AdsorbIN!")
                    # sys.exit(0)

            # Check if "bond distance" is not negative
            if float(adsorb_in["adsorb_distance"]) < 0:
                print("Bond distance illegal in AdsorbIN!")
                sys.exit(0)
    else:  # generate a "AdsorbIN" template
        adsorbin_template = """\
##################+++ AdsorbKit settings +++##################

################+++ Adsorb Settings +++################
adsorb_style          =  bridge     # 吸附模式：top(hcp/fcc)/bridge/centre/manual
    adsorb_distance   =  2          # 吸附分子与基体间原子最小距离(键长)
    
substrate      =  POSCAR-sub        # 基体文件
    substrate_ref     =  1,2        # 基体的参考点, 或手动设定坐标
adsorbate      =  POSCAR-ads        # 含吸附分子的文件
    adsorbate_select  =  all        # 算作吸附分子的原子      
    adsorbate_ref     =  1          # 吸附分子的参考点(选2个则为中点) 

auto_offset           =  True       # 吸附分子与基体最小距离超出(键长 ± 0.1 Å)范围时将吸附分子延Z轴偏移

################+++ Output Settings +++################
output_filename       =  POSCAR     # 输出文件名
coordinate            =  Cartesian  # 坐标系
selective_dynamics    =  False       # 选择性优化
    atoms_to_release  =  3-5        # 放开的原子(按先substrate、后adsorbate排序)
"""
        with open("AdsorbIN", mode="w", encoding="utf-8") as tempfile:
            tempfile.write(adsorbin_template)
        print("\"AdsorbIN\" not found! Template generated!")
        sys.exit(0)
    return adsorb_in


def read_poscar(poscarfile):
    """
    Organise POSCAR file into a dictionary
    :param poscarfile:
    :return:
    """
    if os.path.exists(poscarfile):
        # Read POSCAR
        with open(poscarfile, mode="r", encoding="utf-8") as posf:
            poscar_in = []
            for line in [line.strip() for line in posf]:
                poscar_in.append(line)

        # Sort lines in POSCAR into dictionary
        poscar = {"comment": poscar_in[0], "scale": float(poscar_in[1])}
        if not poscar["scale"] == 1:
            print(f"Warning! Non-one scale factor found at {poscarfile}!")
        # read 3 lattice vectors
        poscar["lattice_vector"] = [[float(y) for y in x] for x in [vector.split() for vector in poscar_in[2:5]]]  # convert strings to floats
        # generate atom list
        atom_list = []  # organise atoms in POSCAR
        for index in range(len(poscar_in[5].split())):
            atom_name = poscar_in[5].split()[index]
            num = poscar_in[6].split()[index]
            [atom_list.append(atom_name) for count in range(int(num))]
        poscar["atom_list"] = atom_list

        # judge if selective dynamics is activated
        if poscar_in[7].lower().startswith("s"):
            poscar["selective"] = True
            coord_start = 9  # where the coordinates starts
        else:
            poscar["selective"] = False
            coord_start = 8
        # Judge Cartesian or Direct, transfer if in Direct
        poscar["coordinate"] = "cartesian"
        atom_position_list = []
        for line in poscar_in[coord_start:coord_start + len(atom_list)]:
            line = [float(x) for x in line.split()[0:3]]
            if poscar_in[coord_start - 1].lower().startswith("c"):
                atom_position_list.append(line)
            else:  # transfer Direct to Cartesian
                # poscar["coordinate"] = "direct"
                atom_position_list.append(d_c_transfer(line, poscar["lattice_vector"]))

        # Adding atom_positions, atom_selective (T/F) and atom velocity
        # Atom index starting from "1" here !
        atom_position = {}
        atom_selective = {}
        # atom_velocity = {}
        for atom_index in range(len(atom_list)):  # atom_index starting from 0
            atom_position[atom_index + 1] = atom_position_list[atom_index]
            if poscar["selective"]:  # selective dynamics on
                atom_selective[atom_index + 1] = poscar_in[coord_start + atom_index].split()[3:6]
            else:  # selective dynamics off (all atoms free)
                atom_selective[atom_index + 1] = ["T", "T", "T"]
            # atom_velocity[atom_index + 1] = poscar_in[coord_start + len(atom_list) + atom_index].split()  # not necessary at this moment
        poscar["atom_position"] = atom_position
        poscar["atom_selective"] = atom_selective
        # poscar["atom_velocity"] = atom_velocity  # Adding atom velocities

    else:
        print(f"{poscarfile} not found!")
        sys.exit(0)

    # comment/scale/lattice_vector/atom_list/selective dynamics/coordinate/(velocity)
    return poscar


def d_c_transfer(coordinate_in, lattice_vector, transfer_mode="d2c"):
    """
    Direct-Cartesian transfer
    :param coordinate_in: list containing 3 floats
    :param lattice_vector:
    :param transfer_mode: d2c or c2d (default: Direct to Cartesian)
    :return: coordinate_out  # list containing 3 floats
    """
    lattice_vector = np.array(lattice_vector).transpose()
    coordinate_mat = np.array(coordinate_in).transpose()
    # Direct to Cartesian transfer
    if transfer_mode == "d2c":
        coordinate_out = np.dot(lattice_vector, coordinate_mat).tolist()
    # Cartesian to Direct transfer
    elif transfer_mode == "c2d":
        lattice_vector_inverse = np.linalg.inv(lattice_vector)  # inverse of lattice vector matrix
        coordinate_out = np.dot(lattice_vector_inverse, coordinate_mat).tolist()
    return coordinate_out


def adsorb_position():
    """
    Determine adsorption position
    :return: adsorb_coordinate
    """
    # "manual" mode
    if adsorbin_para["adsorb_style"] == "manual":
        adsorb_coordinate = [float(i) for i in adsorbin_para["substrate_ref"].split(",")]

    # "top" ("hcp" or "fcc") mode
    elif adsorbin_para["adsorb_style"].lower() == "top":
        substrate_atom_select = translate_atoms(adsorbin_para["substrate_ref"], poscar_substrate["atom_list"], 1)  # starting from "1" !
        # judge if only one atom is selected
        if not len(substrate_atom_select) == 1:
            print("Please include ONE only atom in the invoked adsorb style!")
            sys.exit(1)

        # read atom coordinate
        adsorb_coordinate = copy.copy(poscar_substrate["atom_position"][substrate_atom_select[0]])

        # calculate adsorb position (start from a position high above the site, then decrease z coordinate while calculating its distance from other atoms until smaller than desired bond distance)
        adsorb_coordinate[2] += 10  # coordinate offset
        # Judge distance between adsorb site and other atoms (of the substrate)
        while adsorb_coordinate[2] >= 0:
            for atom in poscar_substrate["atom_position"]:
                distance = calc_distance(poscar_substrate["atom_position"][atom], adsorb_coordinate)
                if distance < float(adsorbin_para["adsorb_distance"]):
                    return adsorb_coordinate
            adsorb_coordinate[2] -= 0.02  # step length of each move

        return adsorb_coordinate  # a list containing three coordinates

    # "bridge" or "centre" mode
    elif adsorbin_para["adsorb_style"].lower() in ["bridge", "centre", "center"]:
        substrate_atom_select = translate_atoms(adsorbin_para["substrate_ref"], poscar_substrate["atom_list"], 1)  # starting from "1" !

        # judge if enough atoms selected
        if not len(substrate_atom_select) >= 2:
            print("Not enough atoms selected in substrate for selected adsorption mode!")
            sys.exit(1)

        # Read coordinates of each atom nominated and calculate centre coordinate
        point_list = [poscar_substrate["atom_position"][substrate_atom_select[i]] for i in range(len(substrate_atom_select))]
        x_avrg = (sum([i[0] for i in point_list])) / len(substrate_atom_select)
        y_avrg = (sum([i[1] for i in point_list])) / len(substrate_atom_select)
        z_avrg = (sum([i[2] for i in point_list])) / len(substrate_atom_select)
        centre = [x_avrg, y_avrg, z_avrg]

        # Determine adsorption site coordinate
        start_position = copy.copy(centre)
        start_position[2] += 10  # offset z-coordinate by 10 Å
        while True:
            if start_position[2] >= 0:
                for atom in substrate_atom_select:
                    distance = calc_distance(start_position, poscar_substrate["atom_position"][atom])
                    if distance <= float(adsorbin_para["adsorb_distance"]):
                        return start_position
                start_position[2] -= 0.02

            # Unable to find until z-coordinate smaller than 0
            else:
                print("Unable to locate adsorb position under desired bond length! Taking centre instead!")
                return centre

    # Unknown adsorption mode
    else:
        print("Unknown adsorption style in adsorbIN.")
        sys.exit(0)


def collect_adsorbate(adsorb_origin):
    """
    Collect and rearrange adsorbate coordinates according to "AdsorbIN"
    :param adsorb_origin: the origin point on which adsorbate is put on the substrate
    :return:
    """
    # Read atom indexes of "adsorbate"
    atom_select = adsorbin_para["adsorbate_select"]  # atom specified in "AdsorbIN" (starting from 1)
    adsorbate_atom_list = poscar_adsorbate["atom_list"]
    atom_index = translate_atoms(atom_select, adsorbate_atom_list)  # starting from 0!

    # Create a atom list in atom types, eg. [H H O]
    atom_list = []
    [atom_list.append(adsorbate_atom_list[index]) for index in atom_index]

    # Determine reference point of the adsorbate
    adsorbin_ref = adsorbin_para["adsorbate_ref"]  # start from 1
    ads_ref_index = translate_atoms(adsorbin_ref, adsorbate_atom_list, 1)

    # judge if the reference point is within the adsorbate
    for atom in ads_ref_index:
        if (atom - 1) not in atom_index:
            print("Selected adsorbate reference point out of range!")
            sys.exit(1)

    # One reference point
    if len(ads_ref_index) == 1:
        adsorbate_ref = copy.deepcopy(poscar_adsorbate["atom_position"][ads_ref_index[0]])  # MUST deepcopy!
    # Two reference points (midpoint)
    elif len(ads_ref_index) == 2:
        ref1 = poscar_adsorbate["atom_position"][ads_ref_index[0]]
        ref2 = poscar_adsorbate["atom_position"][ads_ref_index[1]]
        adsorbate_ref = [(ref1[i] + ref2[i]) / 2 for i in range(0, 3)]
    else:
        print("Unsupported adsorption reference point!")
        sys.exit(1)

    # Read adsorbate coordinates, subtract reference point and add adsorption site coordinate
    coordinate = {}  # starting from 0
    for index in atom_index:
        coordinate[index] = poscar_adsorbate["atom_position"][index + 1]
        for component in range(0, 3):
            coordinate[index][component] = poscar_adsorbate["atom_position"][index + 1][component] - adsorbate_ref[component] + adsorb_origin[component]

    return {"atom_list": atom_list, "atom_position": coordinate}


def combine_poscar(poscar_substrate, adsorbate, adsorbin):
    """
    Combine POSCARs of substrate and adsorbate
    :param poscar_substrate:
    :param adsorbate:
    :return:
    """
    # Create output POSCAR dict
    poscar_output = {}
    poscar_output["comment"] = poscar_substrate["comment"]
    poscar_output["scale"] = poscar_substrate["scale"]
    poscar_output["lattice_vector"] = poscar_substrate["lattice_vector"]
    poscar_output["coordinate"] = adsorbin_para["coordinate"]
    poscar_output["selective"] = adsorbin_para["selective_dynamics"]

    # Separate atom list into atom types and atom numbers
    output_atom_list = poscar_substrate["atom_list"] + adsorbate["atom_list"]
    atom_type = []
    atom_number = []
    start_ele = output_atom_list[0]
    count = 0
    for i in range(len(output_atom_list)):
        if start_ele == output_atom_list[i]:
            count += 1
        else:
            atom_type.append(start_ele)
            atom_number.append(count)
            start_ele = output_atom_list[i]
            count = 1
    atom_type.append(start_ele)
    atom_number.append(count)
    poscar_output["atom_type"] = atom_type
    poscar_output["atom_number"] = atom_number

    # Combine substrate and adsorbate coordinates
    poscar_output["atom_position"] = copy.copy(poscar_substrate["atom_position"])
    adsorbate_list = [adsorbate["atom_position"][atom] for atom in adsorbate["atom_position"]]
    substrate_length = len(poscar_substrate["atom_position"])  # length changes during iteration
    for i in range(len(adsorbate_list)):
        poscar_output["atom_position"][substrate_length + 1 + i] = adsorbate_list[i]

    # Calculate minimum distance from each atom in adsorbate to those in substrate
    distance_min = 10000
    for adsorbate_atom in range(len(poscar_substrate["atom_position"]) + 1, len(poscar_substrate["atom_position"]) + len(adsorbate["atom_position"]) + 1):
        for substrate_atom in poscar_substrate["atom_position"]:
            distance = calc_distance(poscar_output["atom_position"][adsorbate_atom], poscar_output["atom_position"][substrate_atom])
            if distance < distance_min:
                distance_min = distance
    # warn user if distance out of (bond length +- 0.2 A range)
    if (distance_min < float(adsorbin["adsorb_distance"]) - 0.1) or (distance_min > float(adsorbin["adsorb_distance"]) + 0.2):
        pwd = os.path.abspath(".").split("/")[-1]  # works in Linux
        print("\033[33mWarning! Minimum distance of", "%.4f" % distance_min, chr(197), f"between adsorbate and substrate detected in \"{pwd}\"!\033[0m")

    # Auto offset adsorbate along Z-axis
    if adsorbin["auto_offset"].lower().startswith("t"):
        offset_distance = 0
        # towards positive
        if distance_min < (float(adsorbin["adsorb_distance"]) - 0.1):
            while distance_min < float(adsorbin["adsorb_distance"]):
                offset_distance += 0.01
            # offset each atom in adsorbate and recalculate minimum distance
                new_min = 1000  # new minimum distance in each move
                # loop across all atoms in "adsorbate"
                for adsorb_atom in range(len(poscar_substrate["atom_position"]) + 1, len(poscar_substrate["atom_position"]) + len(adsorbate["atom_position"]) + 1):
                    poscar_output["atom_position"][adsorb_atom][2] += 0.01  # offset Z by 0.01 Å
                    # loop across all atoms in "substrate"
                    for atom in poscar_substrate["atom_position"]:
                        dist = calc_distance(poscar_output["atom_position"][adsorb_atom], poscar_output["atom_position"][atom])
                        if dist < new_min:
                            new_min = dist
                distance_min = new_min
            print(f"Caution! Adsorbate offset along Z-axis by ", "%.2f" % offset_distance, " ", chr(197), "!", sep="")

    # towards negative
        elif distance_min > (float(adsorbin["adsorb_distance"]) + 0.2):
            while distance_min > float(adsorbin["adsorb_distance"]):
                offset_distance -= 0.01
            # offset each atom in adsorbate and recalculate minimum distance
                new_min = 1000  # new minimum distance in each move
                # loop across all atoms in "adsorbate"
                for adsorb_atom in range(len(poscar_substrate["atom_position"]) + 1, len(poscar_substrate["atom_position"]) + len(adsorbate["atom_position"]) + 1):
                    poscar_output["atom_position"][adsorb_atom][2] -= 0.01  # offset Z by 0.01 Å
                    # loop across all atoms in "substrate"
                    for atom in poscar_substrate["atom_position"]:
                        dist = calc_distance(poscar_output["atom_position"][adsorb_atom], poscar_output["atom_position"][atom])
                        if dist < new_min:
                            new_min = dist
                distance_min = new_min
            print(f"Caution! Adsorbate offset along Z-axis by ", "%.2f" % offset_distance, " ", chr(197), "!", sep="")

    # Adding selective dynamics info
    if adsorbin_para["selective_dynamics"].lower().startswith("t"):
        selective_info = {}
        atoms_to_release = translate_atoms(adsorbin_para["atoms_to_release"], poscar_substrate["atom_list"] + adsorbate["atom_list"], index_start=1)
        for atom_i in range(1, len(output_atom_list) + 1):
            if atom_i in atoms_to_release:
                selective_info[atom_i] = ["T", "T", "T"]
            else:
                selective_info[atom_i] = ["F", "F", "F"]
        poscar_output["selective_info"] = selective_info

    return poscar_output


def rearrange_poscar(poscar_dict, coordinate, selective):
    """
    Rearrange POSCAR for output
    :param poscar_dict:
    :param coordinate: Cartesian or Direct
    :param selective:
    :return:
    """
    # reformat coordinates
    def format_coord(oldcoord):
        newcoord = []
        for component in range(3):
            if oldcoord[component] >= 0:
                newcoord.append("+" + "%.10f" % oldcoord[component])
            else:
                newcoord.append("%.10f" % oldcoord[component])
        return newcoord

    outputfile = []
    # adding comment, vector scale and lattice vectors to the output POSCAR file
    outputfile.append(str(poscar_dict["comment"]))
    outputfile.append(str(poscar_dict["scale"]))
    for vector in range(3):  # lattice vectors
        outputfile.append("  ".join(format_coord(poscar_dict["lattice_vector"][vector])))

    # Add atom type and counts
    atom_list = []
    for index in range(len(poscar_dict["atom_type"])):
        atom_name = poscar_dict["atom_type"][index]
        num = poscar_dict["atom_number"][index]
        [atom_list.append(atom_name) for count in range(int(num))]

    # Rearrange atom types and numbers
    atom_rearrange = {}
    for key in range(len(atom_list)):
        atom_rearrange[key + 1] = atom_list[key]
    if auto_poscar_rearrange:  # auto-rearrange could be turned on/off here
        after_rearrange_dict = dict(sorted(atom_rearrange.items(), key=lambda kv: (kv[1], kv[0])))  # rearrange dictionary according to value
        print("\033[33mCaution! POSCAR auto rearranged!\033[0m")
    else:
        after_rearrange_dict = atom_rearrange
        print("\033[33mCaution! POSCAR NOT auto rearranged!\033[0m")

    # Regenerate atom types and atom numbers from atom list
    after_rearrange_list = [after_rearrange_dict[key] for key in after_rearrange_dict]
    atom_type = []
    atom_number = []
    start_ele = after_rearrange_list[0]
    count = 0
    for i in range(len(after_rearrange_list)):
        if start_ele == after_rearrange_list[i]:
            count += 1
        else:
            atom_type.append(start_ele)
            atom_number.append(count)
            start_ele = after_rearrange_list[i]
            count = 1
    atom_type.append(start_ele)
    atom_number.append(count)

    outputfile.append("  ".join(atom_type))
    outputfile.append("  ".join([str(i) for i in atom_number]))

    if selective.lower().startswith("t"):
        outputfile.append("Selective Dynamics")

    if coordinate.lower().startswith("c"):
        outputfile.append("Cartesian")
    elif coordinate.lower().startswith("d"):
        outputfile.append("Direct")

    # Rearrange and output atom positions
    for index in after_rearrange_dict:

        # coordinate in Cartesian or Direct
        if adsorbin_para["coordinate"].lower().startswith("d"):
            poscar_dict["atom_position"][index] = d_c_transfer(poscar_dict["atom_position"][index], poscar_substrate["lattice_vector"], "c2d")

        # selective dynamics
        if selective.lower().startswith("t"):
            outline = format_coord(poscar_dict["atom_position"][index]) + poscar_dict["selective_info"][index]
        else:
            outline = format_coord(poscar_dict["atom_position"][index])
        outputfile.append("  ".join(outline))

    return outputfile


if __name__ == "__main__":
    # Read parameters from "AdsorbIN" file
    adsorbin_para = read_adsorbin()
    # Read POSCARs for adsorbate and substrate
    poscar_adsorbate = read_poscar(adsorbin_para["adsorbate"])
    poscar_substrate = read_poscar(adsorbin_para["substrate"])
    if not poscar_adsorbate["scale"] == poscar_substrate["scale"]:
        print("\033[31mERROR! Mismatch in scale factors detected.\033[0m")
        sys.exit(0)

    # Collect and rearrange adsorbate according to "AdsorbIN"
    adsorbate = collect_adsorbate(adsorb_position())

    # Generate output POSCAR file
    poscar_output = rearrange_poscar(combine_poscar(poscar_substrate, adsorbate, adsorbin_para), adsorbin_para["coordinate"], adsorbin_para["selective_dynamics"])

    # Write output POSCAR file
    with open(adsorbin_para["output_filename"], mode="w", encoding="utf-8") as outputf:
        for line in poscar_output:
            outputf.write(line + "\n")

    # Job finish indicator
    pwd = os.path.abspath(".").split("/")[-1]
    print(f"\033[32mPOSCAR generated successfully in \"{pwd}\"!\033[0m")
