#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


class dosLoader:
    def __init__(self, dos_path, dos_filename, append_adsorbate=True) -> None:
        # Load DOS
        assert dos_path.is_dir()
        self.__load_dos(dos_path, dos_filename)


        # Append adsorbate DOS
        if append_adsorbate:
            pass


        # return


    def __append_adsorbate_dos(self):
        pass


    def __load_dos(self, dos_path, dos_filename):
        """Load DOS based on path and DOS filename.

        Args:
            dos_path (Path): DOS storage dir path.
            dos_filename (str): DOS storage file name.

        Raises:
            FileNotFoundError: if not matched DOS file found.

        Returns:
            dict: loaded DOS in dict.

        """
        # Find matched folders
        folders = [i.parent for i in dos_path.glob(f"**/{dos_filename}")]
        if not folders:
            raise FileNotFoundError(f"No match found for DOS file with name {dos_filename}.")


        # Load all matched DOS files
        loaded_dos = {}
        for folder in folders:
            loaded_dos[folder.parts[-1]] = np.load(folder / dos_filename)

        return loaded_dos


# Test area
if __name__ == "__main__":
    from pathlib import Path
    loader = dosLoader(
        dos_path=Path("../../1-perturbation-analysis") / "data" / "1-doping" / "1-perturbed",
        dos_filename="dos_up_71.npy",
        )
