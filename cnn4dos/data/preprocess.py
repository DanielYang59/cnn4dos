"""Preprocess utils.

TODO:
- Avoid hard coding in remove_ghost_state (energy axis selection)
- Make sure no copies are made during "remove_ghost_states".
"""

from cnn4dos.data.edos import Edos


def remove_ghost_states(
    edos: Edos,
    width: int = 1,
    axis_label: str = "energy",
) -> Edos:
    """Remove ghost states from eDOS array, along energy axis.

    During eDOS calculations, we observed that in some cases,
    there is a spike occurring exclusively at the 0th point of
    the entire eDOS along energy axis. We verified the unreality of such
    spike by sliding the eDOS energy windows; while the original spike
    disappears, a new spike emerges at the new 0th position.

    Although the exact cause of this phenomenon remains unknown,
    we decided to remove it anyway. Because its presence would yield data
    preprocessing tricky such as normalization and standardization.

    Parameters:
    - width (int, optional): The width along energy axis to remove
        ghost states. Defaults to 1.

    Raises:
    - ValueError: If the provided axis index is invalid or if the removing
        width is not a positive integer or exceeds the total eDOS width.
    """

    # Get axis index
    e_axis = edos.axes.index(axis_label)

    # Check arg: width
    if not isinstance(width, int) or width <= 0:
        raise ValueError("Removing width should be a positive integer.")
    elif width > edos.edos_arr.shape[e_axis]:
        raise ValueError("Removing width greater than total width.")

    # Remove ghost states by setting corresponding values to zero
    if e_axis == 0:
        edos.edos_arr[:width, :, :] = 0.0
    elif e_axis == 1:
        edos.edos_arr[:, :width, :] = 0.0
    elif e_axis == 2:
        edos.edos_arr[:, :, :width] = 0.0
    else:
        raise ValueError("Axis index out of range.")

    return edos
