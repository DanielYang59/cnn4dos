centre_atom = 71
source_POSCAR_dir = "0-original-Cr-vac-graphene"
tolerance = 0.2


import os
import copy
from matplotlib import cm
import matplotlib.pyplot as plt
from lib.poscar import POSCAR


if __name__ == "__main__":
    # Load poscar
    poscar = POSCAR()
    poscar.read(os.path.join(source_POSCAR_dir, "POSCAR"))

    # Load all atom X/Y coordinates
    poscar.set_coordinate("cartesian")
    coordinates = copy.copy(poscar.ion_positions)
    coordinates = [[float(j) for j in i[:2]] for i in coordinates]

    # Plot lattice boundaries
    lattice_a = poscar.lattice[0]
    lattice_b = poscar.lattice[1]
    point_a_b = lattice_a + lattice_b
    plt.plot([0, lattice_a[0]], [0, lattice_a[1]], color="black", linewidth=4)
    plt.plot([0, lattice_b[0]], [0, lattice_b[1]], color="black", linewidth=4)
    plt.plot([lattice_a[0], point_a_b[0]], [lattice_a[1], point_a_b[1]], color="black", linewidth=4)
    plt.plot([lattice_b[0], point_a_b[0]], [lattice_b[1], point_a_b[1]], color="black", linewidth=4)



    # Plot centre atom
    centre_atom_coord = [float(i) for i in poscar.ion_positions[centre_atom - 1][:2]]

    plt.plot(centre_atom_coord[0], centre_atom_coord[1], marker="o", markersize=15, markerfacecolor="red", markeredgecolor="black")


    # Calculate distance
    distances_dict = {}
    for index in range(len(poscar.ion_positions)):
        if index != (centre_atom - 1):  # skip centre atom
            # calculate distance
            distances_dict[index] = poscar.calculate_distance([centre_atom - 1, index])
    # Sort distance dictionary by distance
    distances_dict = dict(sorted(distances_dict.items(), key=lambda item: item[1]))


    # Group satellites by distance
    grouped_distances = {}
    grouped_indexes = {}
    group_count = 0
    for i in range(len(distances_dict) - 1):
        index_1 = list(distances_dict.keys())[i]
        index_2 = list(distances_dict.keys())[i + 1]
        sat_1 = distances_dict[index_1]
        sat_2 = distances_dict[index_2]

        if (sat_2 - sat_1) <= tolerance:
            if group_count == 0:
                grouped_distances[1] = [sat_1, sat_2]
                grouped_indexes[1] = [index_1, index_2]
                group_count += 1
            else:
                try:
                    grouped_distances[group_count].extend([sat_1, sat_2])
                    grouped_indexes[group_count].extend([index_1, index_2])
                except KeyError:
                    grouped_distances[group_count] = [sat_1, sat_2]
                    grouped_indexes[group_count] = [index_1, index_2]

        else:
            group_count += 1

    # remove redundant atoms
    for group_index in grouped_distances.keys():
        grouped_distances[group_index] = set(grouped_distances[group_index])
        grouped_indexes[group_index] = set(grouped_indexes[group_index])

    print(f"A total of {len(grouped_distances)} groups found.")

    # Generate colormap for plotting
    color_map = cm.get_cmap("gist_ncar", len(grouped_distances))

    # Plot dot by group
    for i, members in grouped_indexes.items():
        # Generate color of group
        color = color_map(i)

        # Plot members of each group with the same color
        for dot in members:
            coord = coordinates[dot]
            plt.plot(coord[0], coord[1], marker="o", markersize=10, markerfacecolor=color, markeredgecolor="black")

            # Add atom index for each atom
            plt.text(coord[0] + 0.1, coord[1] + 0.1,
                     dot + 1, fontsize=12)

        # Print a unique member in each group
        print(f"Group {i}: {list(members)[0] + 1}")


    plt.tight_layout()
    plt.savefig("grouped_atom.png", dpi=300)
    plt.show()
