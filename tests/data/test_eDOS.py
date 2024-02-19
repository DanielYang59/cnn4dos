"""Pytest unit test for eDOS class."""

import re
from pathlib import Path

import numpy as np
import pytest

from cnn4dos.data.eDOS import eDOS
from cnn4dos.utils import ROOT_DIR

# Define test variables
test_data_dir = ROOT_DIR / "tests" / "test_data" / "hea_edos_spin_polarized"

atoms = [0, 15]
orbitals = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2"]
spins = ["up", "down"]
nedos = 100

expected_shape = (len(atoms), len(orbitals), len(spins), nedos)


# Test starts
@pytest.mark.filterwarnings(  # no POTCAR included in test data
    "ignore", match="No POTCAR file with matching TITEL fields was found"
)
class Test_eDOS:
    """Pytest unit test for eDOS class."""

    def test_from_array_and_to_array(self) -> None:
        # Create test numpy array with expected shape
        test_arr = np.random.rand(*expected_shape)
        test_filename = test_data_dir / ".temp_test_arr.npy"

        edos = eDOS()
        edos.array = np.copy(test_arr)

        if test_filename.is_file():
            raise FileExistsError("Test array file already exists.")

        else:
            edos.to_array(test_filename)
            edos.array = np.nan
            edos.from_array(test_filename)
            Path.unlink(test_filename, missing_ok=False)

            assert np.array_equal(edos.array, test_arr)
            # Ensure these two arrays does not share momory location
            assert not np.shares_memory(edos.array, test_arr)

    def test_from_vasprun_success(self) -> None:
        edos = eDOS()

        edos.from_vasprun(
            filename=test_data_dir / "vasprun.xml",
            atoms=atoms,
            orbitals=orbitals,
            spins=spins,
        )

        assert edos.array.shape == expected_shape

    @pytest.mark.parametrize(
        "invalid_atoms, expected_error",
        [
            ([-1, 0], "atom index must be int in range [0, 15]"),
            (["Fe", 1], "atom index must be int in range [0, 15]"),
            ([1, 1], "Duplicate atoms not allowed."),
        ],
    )
    def test_from_vasprun_invalid_atoms(self, invalid_atoms, expected_error) -> None:
        edos = eDOS()

        with pytest.raises(ValueError, match=re.escape(expected_error)):
            edos.from_vasprun(
                filename=test_data_dir / "vasprun.xml",
                atoms=invalid_atoms,
                orbitals=["s", "px"],
                spins=["up", "down"],
            )

    @pytest.mark.parametrize(
        "invalid_orbitals, expected_error",
        [
            ([-1, 0], "Invalid orbital found."),
            (["s", 1], "Invalid orbital found."),
            (["s", "s"], "Duplicate orbitals not allowed."),
        ],
    )
    def test_from_vasprun_invalid_orbitals(
        self, invalid_orbitals, expected_error
    ) -> None:
        edos = eDOS()

        with pytest.raises(ValueError, match=expected_error):
            edos.from_vasprun(
                filename=test_data_dir / "vasprun.xml",
                atoms=[0, 15],
                orbitals=invalid_orbitals,
                spins=["up", "down"],
            )

    @pytest.mark.parametrize(
        "invalid_spins, expected_error",
        [
            ([-1, 0], "spin must be either 'up' or 'down'."),
            (["up", "up"], "Duplicate spins not allowed."),
        ],
    )
    def test_from_vasprun_invalid_spins(self, invalid_spins, expected_error) -> None:
        edos = eDOS()

        with pytest.raises(ValueError, match=expected_error):
            edos.from_vasprun(
                filename=test_data_dir / "vasprun.xml",
                atoms=[0, 15],
                orbitals=["s", "px"],
                spins=invalid_spins,
            )

    def test_swap_axes(self) -> None:
        original_arr = np.random.rand(*expected_shape)
        edos = eDOS()
        edos.array = np.copy(original_arr)

        edos.swap_axes(0, 2)

        assert not np.shares_memory(edos.array, original_arr)
        assert np.array_equal(edos.array, np.swapaxes(original_arr, 0, 2))
