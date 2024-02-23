"""Pytest unit test for Edos class."""

import re
from pathlib import Path

import numpy as np
import pytest

from cnn4dos.data import Edos
from cnn4dos.utils import ROOT_DIR


@pytest.mark.filterwarnings(  # no POTCAR included in test data
    "ignore: No POTCAR file with matching TITEL fields was found"
)
class Test_Edos:
    """Pytest unit test for Edos class."""

    test_data_dir = (
        ROOT_DIR / "tests" / "test_data" / "hea_edos_spin_polarized"
    )

    atoms = [0, 15]
    orbitals = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2"]
    spins = ["up", "down"]
    nedos = 100

    expected_shape = (len(atoms), len(orbitals), len(spins), nedos)

    @pytest.mark.parametrize(
        "edos_arr, expected_shape, axes",
        [
            (
                np.ones((3, 4, 5, 6)),
                (3, 4, 5, 6),
                ("orbital", "energy", "atom", "spin"),
            ),
            (np.ones((3, 4, 5, 6)), (3, 4, 5, 6), None),
            (np.ones((3, 4, 5, 6)), None, None),
        ],
    )
    def test_valid_init(self, edos_arr, expected_shape, axes) -> None:
        Edos(edos_arr, expected_shape, axes)

    @pytest.mark.parametrize(
        "edos_arr, expected_shape, axes, error_msg",
        [
            (
                np.ones((1, 2, 3)),
                None,
                ("orbital", "energy", "atom", "spin"),
                "Expect a 4D eDOS numpy array.",
            ),
            (
                [1, 2, 3, 4],
                None,
                ("orbital", "energy", "atom", "spin"),
                "Expect a 4D eDOS numpy array.",
            ),
            (
                None,
                (1, 2, 3),
                ("orbital", "energy", "atom", "spin"),
                "Expect a 4D shape.",
            ),
            (
                None,
                None,
                ("orbital", "energy", "atom"),
                "Axes should be tuple with only orbital, energy, atom, spin.",
            ),
            (
                None,
                None,
                ("h", "e", "ll", "o"),
                "Axes should be tuple with only orbital, energy, atom, spin.",
            ),
            (
                None,
                None,
                ("atom", "energy", "atom", "spin"),
                "Axes should be tuple with only orbital, energy, atom, spin.",
            ),
        ],
    )
    def test_invalid_init(
        self, edos_arr, expected_shape, axes, error_msg
    ) -> None:
        with pytest.raises(ValueError, match=error_msg):
            Edos(edos_arr=edos_arr, expected_shape=expected_shape, axes=axes)

    def test_from_array_and_to_array_file(self) -> None:
        # Create test numpy array with expected shape
        test_arr = np.ones(self.expected_shape)
        test_filename = self.test_data_dir / ".temp_test_arr.npy"

        edos = Edos()
        edos.edos_arr = np.copy(test_arr)

        if test_filename.is_file():
            raise FileExistsError("Test array file already exists.")

        else:
            edos.to_array(test_filename)
            edos.edos_arr = np.nan
            edos.from_arr_file(test_filename)
            Path.unlink(test_filename, missing_ok=False)

            assert np.array_equal(edos.edos_arr, test_arr)
            # Ensure these two arrays does not share momory location
            assert not np.shares_memory(edos.edos_arr, test_arr)

    def test_from_vasprun_success(self) -> None:
        edos = Edos()

        edos.from_vasprun(
            filename=self.test_data_dir / "vasprun.xml",
            atoms=self.atoms,
            orbitals=self.orbitals,
            spins=self.spins,
        )

        assert edos.edos_arr.shape == self.expected_shape

    @pytest.mark.parametrize(
        "invalid_atoms, expected_error",
        [
            ([-1, 0], "atom index must be int in range [0, 15]"),
            (["Fe", 1], "atom index must be int in range [0, 15]"),
            ([1, 1], "Duplicate atoms not allowed."),
        ],
    )
    def test_from_vasprun_invalid_atoms(
        self, invalid_atoms, expected_error
    ) -> None:
        edos = Edos()

        with pytest.raises(ValueError, match=re.escape(expected_error)):
            edos.from_vasprun(
                filename=self.test_data_dir / "vasprun.xml",
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
        edos = Edos()

        with pytest.raises(ValueError, match=expected_error):
            edos.from_vasprun(
                filename=self.test_data_dir / "vasprun.xml",
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
    def test_from_vasprun_invalid_spins(
        self, invalid_spins, expected_error
    ) -> None:
        edos = Edos()

        with pytest.raises(ValueError, match=expected_error):
            edos.from_vasprun(
                filename=self.test_data_dir / "vasprun.xml",
                atoms=[0, 15],
                orbitals=["s", "px"],
                spins=invalid_spins,
            )

    def test_reset_axes(self) -> None:
        edos = Edos(
            edos_arr=np.ones((3, 4, 5, 6)),
            axes=("orbital", "energy", "atom", "spin"),
        )

        edos.reset_axes(("spin", "atom", "energy", "orbital"))

        assert edos.edos_arr.shape == (6, 5, 4, 3)  # type: ignore

    def test_reset_axes_same(self) -> None:
        with pytest.warns(UserWarning, match="Old and new axes are the same."):
            edos = Edos(
                edos_arr=np.ones((3, 4, 5, 6)),
                axes=("orbital", "energy", "atom", "spin"),
            )

            edos.reset_axes(("orbital", "energy", "atom", "spin"))

    @pytest.mark.parametrize(
        "invalid_axes",
        [
            ("orbital", "energy", "atom"),
            ("h", "e", "ll", "o"),
            ("orbital", "energy", "atom", "atom"),
        ],
    )
    def test_reset_axes_invalid_axes(self, invalid_axes) -> None:
        edos = Edos(
            edos_arr=np.ones((3, 4, 5, 6)),
            axes=("orbital", "energy", "atom", "spin"),
        )

        with pytest.raises(
            ValueError,
            match="Axes should be tuple with only orbital, energy, atom, spin.",  # noqa: E501
        ):
            edos.reset_axes(invalid_axes)

    def test_remove_ghost_states(self) -> None:
        edos = Edos(
            edos_arr=np.ones((3, 4, 5, 6)),
            axes=("orbital", "energy", "atom", "spin"),
        )

        edos.remove_ghost_states(width=1)

        assert np.all(edos.edos_arr[:, :1, :, :] == 0.0)

    @pytest.mark.parametrize(
        "invalid_width, error_msg",
        [
            (0, "Removing width should be a positive integer."),
            (5, "Removing width greater than total width."),
        ],
    )
    def test_remove_ghost_states_invalid_width(
        self, invalid_width, error_msg
    ) -> None:
        with pytest.raises(ValueError, match=error_msg):
            edos = Edos(
                edos_arr=np.ones((3, 4, 5, 6)),
                axes=("orbital", "energy", "atom", "spin"),
            )

            edos.remove_ghost_states(invalid_width)
