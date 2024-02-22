"""Pytest unit test for util-get_fermi_level."""

import shutil

import pytest

from cnn4dos.utils import ROOT_DIR, get_fermi_level


@pytest.mark.filterwarnings(  # no POTCAR included in test data
    "ignore", match="No POTCAR file with matching TITEL fields was found"
)
class Test_get_fermi_level:
    test_data_dir = ROOT_DIR / "tests" / "test_data" / "fcc_si_dos"
    digits = 4

    def test_from_vasprun(self, tmp_path):
        test_dir = tmp_path / "vasprun_test"
        test_dir.mkdir()

        shutil.copy(self.test_data_dir / "vasprun.xml", test_dir)

        fermi_level = get_fermi_level(test_dir, self.digits)

        assert fermi_level == 9.8951

    def test_from_outcar(self, tmp_path):
        test_dir = tmp_path / "outcar_test"
        test_dir.mkdir()

        shutil.copy(self.test_data_dir / "OUTCAR", test_dir)

        fermi_level = get_fermi_level(test_dir, self.digits)

        assert fermi_level == 9.8951

    def test_from_wavecar(self, tmp_path):
        test_dir = tmp_path / "wavecar_test"
        test_dir.mkdir()

        shutil.copy(self.test_data_dir / "WAVECAR", test_dir)

        fermi_level = get_fermi_level(test_dir, self.digits)

        assert fermi_level == 9.8951

    def test_no_file_avail(self, tmp_path):
        test_dir = tmp_path / "no_file_test"
        test_dir.mkdir()

        with pytest.raises(
            FileNotFoundError,
            match="Cannot find any file containing fermi level.",
        ):
            get_fermi_level(test_dir / "nonexistent_file")
