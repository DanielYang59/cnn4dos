"""Convert various types of atom selection into an atom index list."""

import warnings
from typing import Union


def convert_atom_selection(
    atom_list: list[str],
    selections: list[Union[int, str]],
    sel_index_start: int = 1,
    range_delimiter: str = "-",
    allow_duplicate: bool = False,
) -> list[int]:
    """Convert mixed-typed atom selections to a list of indices.

    Parameters:
        atom_list (list[str]): A list of elements, like ["Fe", "Co", "Ni"].
        selections (list[Union[int, str]]): A list of selections, which can be
            either indices or string representations of atom labels or ranges
            (e.g., '1', 'Fe', '2-5').
        sel_index_start (int, optional): The starting index from
            user selection. Defaults to 1.
        range_delimiter (str, optional): The delimiter used to specify
            a range of atom indices in "2-5". Defaults to '-'.
        allow_duplicate (bool, optional): Whether duplicate indices are
            allowed in the selections. Defaults to False.

    Returns:
        list[int]: A list of indices representing the selected atoms.

    Raises:
        ValueError: If duplicate indices are found in the selections and
            allow_duplicate is False, or if any index in the selections
            is out of range.

    Note:
        The output index starts from 0.
    """

    # Check arguments
    if not (atom_list and all(isinstance(atom, str) for atom in atom_list)):
        raise ValueError("Expect atom list as a list of str.")
    if sel_index_start not in {0, 1}:
        raise ValueError("Index must start from either 0 or 1.")

    # Perform itemwise conversion
    atom_indices = []
    for item in selections:
        # Index directly like 1
        if isinstance(item, int):
            atom_indices.append(item - sel_index_start)

        # Element symbol like "Fe"
        elif isinstance(item, str) and range_delimiter in item:
            matches = [i for i, ele in enumerate(atom_list) if ele == item]
            if not matches:
                raise IndexError(f"Didn't find match for {item}.")
            atom_indices.extend(matches)

        # Index range like "1-10"
        elif isinstance(item, str) and range_delimiter not in item:
            # Split and check range selection
            parts = [int(i) - sel_index_start for i in item.split(range_delimiter)]
            if len(parts) != 2 or parts[0] >= parts[1]:
                raise ValueError(f"Invalid range selection {item}.")

            atom_indices.extend(list(range(parts[0], parts[1] + 1)))

        else:
            raise ValueError("Expect selections as a list of str or int.")

    # Check for duplicate and out-of-bound index
    if len(set(atom_indices)) != len(atom_indices):
        if allow_duplicate:
            warnings.warn("Duplicate found in atom indexes.")
        else:
            raise ValueError("Duplicate not allowed in atom indexes.")

    elif any(index < 0 or index >= len(atom_list) for index in atom_indices):
        raise ValueError("Atom selection out of bound.")

    return atom_indices
