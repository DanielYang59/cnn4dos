"""
TODO: reposition atom in set vacuum method.
TODO: add index list checker decorator.
"""


__author__ = "Haoyu Yang"


import os
import numpy as np


class POSCAR:
    def __init__(self):
        """
        """
        pass

    def __str__(self):
        return "Class for POSCAR from VASP"

    def read(self, path):
        """
        Import and parse POSCAR file.
        """
        # Import POSCAR file
        assert os.path.exists(path), "POSCAR file not found."
        assert os.path.isfile(path)
        self.path = path  # path to POSCAR file
        with open(self.path) as f:
            content = f.readlines()
        
        # Parse blocks of POSCAR file
        ## Comment
        self.comment = content[0].strip()

        ## Scaling factor
        self.scaling_factor = float(content[1].strip())

        ## Lattice
        self.lattice = np.array([i.strip().split() for i in content[2:5]])  
        self.lattice = self.lattice.astype(float)  # NOTE: string to float here 

        ## Ion species names
        self.ion_species = content[5].strip().split()

        ## Ion numbers
        self.ion_numbers = [int(i) for i in content[6].strip().split()]
        assert len(self.ion_species) == len(self.ion_numbers)
        ### Calculate number of ions
        num_of_ion = sum(self.ion_numbers)

        ## Selective dynamics
        if content[7].lower().startswith("s"):
            self.selective_dynamics = True
        else:
            self.selective_dynamics = False

        ## Coordinate
        if self.selective_dynamics:
            coordinate_line_index = 8
        else:
            coordinate_line_index = 7
        if content[coordinate_line_index].lower().startswith("c"):
            self.coordinate = "cartesian"
        elif content[coordinate_line_index].lower().startswith("d"):
            self.coordinate = "direct"
        else:
            raise ValueError("Unknown coordinate system in POSCAR.")

        ## Ion positions
        if self.selective_dynamics:
            self.ion_positions = [i.strip().split()[:3] for i in content[9: 9 + num_of_ion]]
        else:
             self.ion_positions = [i.strip().split()[:3] for i in content[8: 8 + num_of_ion]]
        assert len(self.ion_positions) == num_of_ion

        ## Ion selective dynamics info
        if self.selective_dynamics:
            self.selective_dynamics_info = [i.strip().split()[3:6] for i in content[9: 9 + num_of_ion]]
        else:
            self.selective_dynamics_info = [["T", "T", "T"] for i in range(num_of_ion)]
        assert len(self.selective_dynamics_info) == num_of_ion

        ## Ion velocities 
        if self.selective_dynamics:
            self.velocities = [i.strip() for i in content[10 + num_of_ion: 10 + num_of_ion * 2]]
        else:
            self.velocities = [i.strip() for i in content[9 + num_of_ion: 9 + num_of_ion * 2]]
        ### no velocity info available
        if len(self.velocities)  == 0:
            self.velocities = [["0", "0", "0"] for i in range(num_of_ion)]
        assert len(self.velocities) == num_of_ion

    def write(self, filename="POSCAR"):
        """
        Export POSCAR to file.
        Args:
            filename: output name of file
        """
        # Initialize output list
        output_content = []
        
        ## Comment
        output_content.append(self.comment)
        ## Scaling factor
        output_content.append(str(self.scaling_factor))

        ## Lattice
        for line in self.lattice:
            output_content.append(" ".join([str(j) for j in line.tolist()]))
        
        ## Ion species and numbers
        output_content.append(" ".join(self.ion_species))
        output_content.append(" ".join([str(i) for i in self.ion_numbers]))

        ## Selective dynamics
        if self.selective_dynamics:
            output_content.append("Selective")
        
        ## Coordinate
        output_content.append(self.coordinate.capitalize())

        ## Write ion position and selective dynamics info (if required)
        for index, value in enumerate(self.ion_positions):
            if self.selective_dynamics:
                output_content.append(" ".join(value) + " " + " ".join(self.selective_dynamics_info[index]))

            else:
                output_content.append(" ".join(value))

        # Write to file
        with open(filename, mode="w") as f:
            f.write("\n".join(output_content))

    def get_element_list(self):
        """
        Get elements as a list.
        Returns: 
            element_list, like ["C", "N", "N", "O", "O", "O"].
        """
        element_list = []
        for index, element in enumerate(self.ion_species):
            number = self.ion_numbers[index]
            element_list.extend([element] * number)

        return element_list

    def update_with_element_list(self, new_element_list):
        """
        Update POSCAR with given element list.
        
        Args:
            element_list: list of elements (from get_element_list method). For example, ["C", "N", "H"]
            
        """
        assert len(new_element_list) > 0
        # Generate new element species and numbers 
        species_list = []
        numbers_list = []
        for index, ele in enumerate(new_element_list):
            # New species or new list
            if index == 0 or species_list[-1] != ele:
                species_list.append(ele)
                numbers_list.append(1)
            # Existing species
            else:
                numbers_list[-1] += 1

        # Update POSCAR attribute
        assert len(new_element_list) == sum(numbers_list)
        self.ion_species = species_list
        self.ion_numbers = numbers_list

    def decode_atom_selection(self, user_input):
        """
        Decode user selection of atom to index list
        Args:
            user_input: user selection of atoms in string, for example "Fe, 1-3, 10" (index starts from 1)
        Returns:
             index_list: list of atom index (index starts from 0)
        """
        # Check user input (should be string)
        assert len(user_input) > 0
        assert isinstance(user_input, str), "User input should be string."
        # Parse user input string to list
        user_input = user_input.replace(" ", "").split(",")

        # Decode each element from user input
        index_list = []
        for element in user_input:
            ## Single number like "1"
            if element.isdigit():
                index_list.append(int(element) - 1)  # offset by 1
            ## Index range like "1-10"
            elif "-" in element:
                index_list.extend(range(int(element.split("-")[0]) - 1, int(element.split("-")[-1])))  # offset by 1
            ## Element symbols like "Fe"
            elif element.isalpha():
                # Match element symbol
                element_list = self.get_element_list()
                matched_index = [index for index, elem in enumerate(element_list) if elem == element]
                index_list.extend(matched_index)

            else:
                raise ValueError(f"Unknown atom selection {element}.")
        
        # Remove duplicate if exists
        if len(index_list) != len(set(index_list)):
            raise ValueError("Duplicate found in atom selection.")

        assert len(index_list) > 0, "No matched atom"
        return index_list
        
    def sort_atom(self, order_index):
        """
        Sort atom order based on given list of indexes.
        Args:
            order_index: sequence of desired atom index
        """
        # Sort ion species and numbers
        element_list = self.get_element_list()
        self.update_with_element_list([element_list[i] for i in order_index])

        # Sort positions, selective dynamic info and velocities
        self.ion_positions = [self.ion_positions[i] for i in order_index]
        self.selective_dynamics_info = [self.selective_dynamics_info[i] for i in order_index]
        self.velocities = [self.velocities[i] for i in order_index]

    def set_coordinate(self, coordinate):
        """
        Set coordinate system of given POSCAR to Direct or Cartesian.
        Args:
            coordinate: target coordinate of POSCAR file, either "cartesian" or "direct"
        """
        # Standardize coordinate selection
        if coordinate.lower().startswith("c"):
            coordinate = "cartesian"
        elif coordinate.lower().startswith("d"):
            coordinate = "direct"
        else:
            raise ValueError("Unknown coordinate. Please choose either Cartesian or Direct.")

        # Only act when target coordinate differs from source coordinate
        if coordinate != self.coordinate:
            # Direct to Cartesian transfer
            if coordinate == "cartesian":
                translation_matrix = self.lattice.transpose()
            # Cartesian to Direct transfer
            else:
                translation_matrix = np.linalg.inv(self.lattice.transpose())  # calculate inverse of lattice
            
            # Apply translation to each ion position
            new_ion_positions = []
            for position in self.ion_positions:
                position = np.array([float(i) for i in position])
                position = np.dot(translation_matrix, position.transpose()).tolist()
                new_ion_positions.append([str(round(i, 10)) for i in position])  # Keep 10 digits
                
            # Update information
            self.ion_positions = new_ion_positions
            self.coordinate = coordinate

    def set_vacuum(self, vacuum):
        """
        Set vacuum level of POSCAR.
        Args:
            vacuum: target vacuum level thickness in Angstrom

        """
        # Set coordinate to Cartesian
        self.coordinate = "cartesian"
        self.set_coordinate(coordinate="cartesian")

        # Check scaling factor
        if self.scaling_factor != 1:
            self.reset_scaling_factor

        # Get lattice vector γ
        gamma = self.lattice[2]
        assert gamma[0] == 0 and gamma[1] == 0
        lattice_thickness = gamma[2]

        # Get old vacuum level thickness
        z_coordinates = [float(i[2]) for i in self.ion_positions]
        z_coordinates.sort()
        old_vacuum = lattice_thickness - (z_coordinates[-1] - z_coordinates[0])
        print(f"Original vacuum thickness is {round(old_vacuum, 2)}. Resize to {vacuum}.")

        # Add difference in vacuum thickness to lattice vector gamma
        self.lattice[2, 2] = lattice_thickness + (vacuum - old_vacuum) 
        assert isinstance(vacuum, (float, int))

    def fix_atom(self, index_list):
        """
        Fix ions based on index from given list, index starts from 0.
        Args:
            index_list: list of atom indexes to fix
        """
        # Check index list
        assert len(index_list) > 0 
        assert all([isinstance(i, int) for i in index_list])

        # Fix ion based on index list
        new_selective_dynamics_info = []
        for index, value in enumerate(self.selective_dynamics_info):
            if index in index_list:
                new_selective_dynamics_info.append(["F", "F", "F"])
            else:
                new_selective_dynamics_info.append(["T", "T", "T"])
        
        # Update info
        self.selective_dynamics_info = new_selective_dynamics_info
        self.selective_dynamics = True
    
    def sort_by_z(self):
        """
        Sort ion positions based on their Z coordinates.
        """
        # Generate Z-coordinate list and sort
        z_coords = [position[2] for position in self.ion_positions]
        index_for_order = sorted(range(len(z_coords)), key=z_coords.__getitem__)  # list of index for new z-coordinates

        # Sort POSCAR
        self.sort_atom(index_for_order)
    
    def reset_scaling_factor(self):
        """
        Reset scaling factor to 1.
        # DEBUG poscar的缩放系数是作用在晶胞参数上还是体积？ # TODO
        """
        if self.scaling_factor != 1:
            print(f"Warning! Currently scaling factor \"{self.scaling_factor}\" would be reset to \"1\".")
            # Calculate new lattice vectors
            self.lattice *= self.scaling_factor
            self.scaling_factor = 1

    def remove_atom(self, index_list):
        """
        Remove atom by atom index list.
        Args:
            index_list: list of atom index to remove
        """
        assert len(index_list) > 0
        # Get element list
        total_ion_num = sum(self.ion_numbers)

        # Update species and number info
        element_list = self.get_element_list()
        new_element_list = [element_list[i] for i in range(total_ion_num) if i not in index_list]
        self.update_with_element_list(new_element_list)

        # Update position, selective dynamics info and velocities
        new_ion_positions = [self.ion_positions[i] for i in range(total_ion_num) if i not in index_list]
        self.ion_positions = new_ion_positions
        
        new_selective_dynamics_info = [self.selective_dynamics_info[i] for i in range(total_ion_num) if i not in index_list]
        self.selective_dynamics_info = new_selective_dynamics_info
        
        new_velocities = [self.velocities[i] for i in range(total_ion_num) if i not in index_list]
        self.velocities = new_velocities

    def replace_atom(self, index_list, new_element):
        """
        Replace element based on given index list.Use decode_atom_selection method to decode input from user.
        Args:
            index_list: list of atom indexes to replace
            new_element: new element symbol
        """
        assert len(index_list) > 0
        assert isinstance(new_element, str)

        # Replace element based on given index list
        old_element_list = self.get_element_list()
        new_element_list = []

        for index, ele in enumerate(old_element_list):
            if index in index_list:
                new_element_list.append(new_element)
            else:
                new_element_list.append(ele)

        # Update POSCAR with element list
        self.update_with_element_list(new_element_list)

    def reposition_all(self, mode, distance=None):
        """
        Reposition all atoms along Z-axis to top/bottom/centre or for a given distance.
        Args:
            mode: working mode of method, should only be "top", "bottom", "centre/center" or "manual"
            distance: distance in Angstrom to move (only required for manual mode)
        Note:
            not working on monoclinic or triclinic systems
        """
        # Get z-coordinate list and lattice thickness
        self.set_coordinate("cartesian")
        z_coords = sorted([float(position[2]) for position in self.ion_positions])
        lattice_thickness = float(self.lattice[2][2])

        # Calculate moving vector
        assert mode in {"top", "bottom", "centre", "center", "manual"}
        if mode == "top":
            distance = lattice_thickness - z_coords[-1]

        elif mode == "bottom":
            distance = -z_coords[0]

        elif mode == "manual":
            distance = distance

        else:  # centre/center
            lattice_centre = 0.5 * lattice_thickness  # Calculate lattice centre
            cluster_centre = sum(z_coords) / len(z_coords)  # Calculate centre of atom cluster
            distance = lattice_centre - cluster_centre  

        # Invoke move_atoms method
        assert isinstance(distance, (float, int)) 
        moving_vector = np.array([0, 0, distance])
        self.move_atoms(index_list=range(len(self.ion_positions)), vector=moving_vector)

    def move_atoms(self, index_list, vector):
        """
        Move atoms along a given vector.
        Args:
            index_list: list of atom indexes to move
            vector: (np.ndarray) move direction in numpy array
        """
        # Use Cartesian coordinate for moving
        if self.coordinate == "direct":
            self.set_coordinate("cartesian")

        # Move atom according to index list
        assert len(index_list) > 0
        assert isinstance(vector, np.ndarray) 
        new_position = []
        for index, position in enumerate(self.ion_positions):
            # Calculate new position
            if index in index_list:
                position = np.array([float(i) for i in position]) + vector
            else:
                position = np.array([float(i) for i in position])

            # Numpy array to str list
            position = [str(round(i, 10)) for i in position]
            new_position.append(position)

        # Update attribute
        self.ion_positions = new_position

    def calculate_distance(self, atom_index_pair):
        """
        Calculate the distance of a given atom pair.
        Args:
            atom_index_pair: (list of int) a pair of atom indexes to calculate distance, index starts from 0
        Returns:
            distance: calculated distance in Angstrom
        """
        assert isinstance(atom_index_pair, list)
        assert len(atom_index_pair) == 2

        # Calculate distance between given atoms
        self.set_coordinate("cartesian")
        try:
            coord_0 = np.array(self.ion_positions[atom_index_pair[0]]).astype(float)
            coord_1 = np.array(self.ion_positions[atom_index_pair[1]]).astype(float)
        except IndexError:
            raise IndexError("Please check your atom index (index starts from 0).")
        return np.linalg.norm(coord_0 - coord_1)

    def calculate_neighbour(self, centre_atom):
        """
        Calculate distance from a given atom and sort by distance.
        Args:
            centre_atom: index of centre atom, starts from 0
        Returns: 
            indexes: sorted list of atom indexes
            distances: sorted list of distances to centre atom
        """
        assert isinstance(centre_atom, int)

        # Calculate distance for all atom pairs
        distances = [self.calculate_distance([index, centre_atom]) for index, _ in enumerate(self.ion_positions)]

        # Rank distance into dictionary
        indexes = sorted(range(len(distances)), key=distances.__getitem__)
        distances = sorted(distances)

        return (indexes, distances)

    def mirror_all_atoms(self, reference_plane_Z=None):
        """Mirror all atoms along X-Y plane, referenced to the centre of all atoms.
        
        Args:
            reference_plane_Z ((int, float)): Z coordinate of reference plane for mirroring
            
        """
        # First calculate the Z coordinate of reference plane
        if reference_plane_Z is None:
            z_coords = [float(position[2]) for position in self.ion_positions]
            reference_plane_Z = sum(z_coords) / len(z_coords) 
        assert isinstance(reference_plane_Z, (int, float))

        # Loop through all atom position and calculate new positions
        positions = self.ion_positions
        new_positions = []
        for i in positions:
            # Calculate distance from reference plane
            distance = float(i[2]) - reference_plane_Z
            # Update with new Z coordinate
            i[2] = str(float(i[2]) - 2*distance)
            new_positions.append(i)
        self.ion_positions = new_positions

    def set_atom_order(self, new_element_list):
        """Update atom 
        
        """
        # Check old and new element list
        old_element_list = self.get_element_list()
        assert set(old_element_list) == set(new_element_list)
        
        # Generate new index list
        new_index_list = []
        for element in new_element_list:
            i = old_element_list.index(element)
            new_index_list.append(i)
            old_element_list[i] = None

        # Update element list
        self.update_with_element_list(new_element_list)
        
        # Update ion positions, velocities and selective dynamics
        self.ion_positions = [self.ion_positions[i] for i in new_index_list]
        self.selective_dynamics_info = [self.selective_dynamics_info[i] for i in new_index_list] 
        self.velocities = [self.velocities[i] for i in new_index_list]
        