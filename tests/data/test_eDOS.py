import re

import pytest

from cnn4dos.data.eDOS import eDOS
from cnn4dos.utils import ROOT_DIR

test_data_dir = ROOT_DIR / "tests" / "test_data"


@pytest.mark.filterwarnings(  # no POTCAR included in test data
    "ignore", match="No POTCAR file with matching TITEL fields was found"
)
class Test_eDOS:
    def test_from_vasprun_success(self) -> None:
        atoms = [0, 15]
        orbitals = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2"]
        spins = ["up", "down"]
        nedos = 100

        edos = eDOS()

        edos.from_vasprun(
            filename=test_data_dir / "edos-spin-polarized" / "vasprun.xml",
            atoms=atoms,
            orbitals=orbitals,
            spins=spins,
        )

        assert edos.array.shape == (len(atoms), len(orbitals), len(spins), nedos)

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
                filename=test_data_dir / "edos-spin-polarized" / "vasprun.xml",
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
                filename=test_data_dir / "edos-spin-polarized" / "vasprun.xml",
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
                filename=test_data_dir / "edos-spin-polarized" / "vasprun.xml",
                atoms=[0, 15],
                orbitals=["s", "px"],
                spins=invalid_spins,
            )
