"""Pytest unit test for util-list_folders."""

from pathlib import Path
from tempfile import TemporaryDirectory

from cnn4dos.utils import list_folders


class Test_list_folders:
    def test_list_folders(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            folders = [
                tmp_path / "folder1",
                tmp_path / "folder2",
                tmp_path / ".hidden_folder",
            ]
            for folder in folders:
                folder.mkdir()

            result = list_folders(tmp_path)
            assert len(result) == 2
            assert all(folder in result for folder in folders[:2])
            assert (tmp_path / ".hidden_folder") not in result

    def test_must_have(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_dir = tmp_path / "test_folder"
            test_dir.mkdir()

            folder1 = test_dir / "folder1"
            folder1.mkdir()

            folder2 = test_dir / "folder2"
            folder2.mkdir()
            (folder2 / "INCAR").touch()
            (folder2 / "POSCAR").touch()

            folders = list_folders(test_dir, must_have=["INCAR", "POSCAR"])
            assert [folder.name for folder in folders]

    def test_show_hidden(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            folders = [
                tmp_path / "folder1",
                tmp_path / "folder2",
                tmp_path / ".hidden_folder",
            ]
            for folder in folders:
                folder.mkdir()

            result = list_folders(tmp_path, ignore_hidden=False)
            assert len(result) == 3
            assert all(folder in result for folder in folders)
