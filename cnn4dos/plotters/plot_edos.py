"""Plot eDOS for quick visualization or inspection,
not intended for publish-quality figure.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_edos_1d(
    edos: np.ndarray, energy: Optional[np.ndarray] = None
) -> None:
    """Plot 1D eDOS.

    Parameters:
    - edos (np.ndarray): 1D eDOS array.
    - energy (Optional[np.ndarray], optional): 1D energy array
        corresponding to the eDOS. If None,
        indices of the edos array will be used. Defaults to None.

    Raises:
    - ValueError: If the provided eDOS array is not one-dimensional
        or if the shapes of the eDOS and energy arrays differ.
    """

    # Check eDOS array type and shape
    if not isinstance(edos, np.ndarray) or edos.ndim != 1:
        raise TypeError("Expect 1D eDOS array.")

    # Generate 1D eDOS plot
    if energy is None:
        plt.plot(edos)

    else:
        # Check eDOS and energy arrays shape
        if edos.shape != energy.shape:
            raise ValueError("eDOS and energy shape differ.")
        plt.plot(energy, edos)

    # Set x/y axis labels
    plt.xlabel("Energy (eV)", fontsize=18)
    plt.ylabel("eDOS (states/eV)", fontsize=18)

    plt.show()
