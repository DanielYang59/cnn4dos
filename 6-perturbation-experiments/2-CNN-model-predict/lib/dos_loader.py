#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd


class dosLoader:
    def __init__(self, dos_path, dos_filename, append_adsorbate=False, adsorbate_dosfile=None, adsorbate_numAtoms=None) -> None:
        # Load DOS in shape (NEDOS, numChannels)
        assert dos_path.is_dir()
        self.__load_dos(dos_path, dos_filename)


        # Append adsorbate DOS if required
        if append_adsorbate:
            self.loaded_dos = self.__append_adsorbate_dos(adsorbate_dosfile, adsorbate_numAtoms)


    def __append_adsorbate_dos(self, adsorbate_dosfile, adsorbate_numAtoms):
        """Append adsorbate DOS array.

        Args:
            adsorbate_dosfile (Path): adsorbate DOS numpy file.
            adsorbate_numAtoms (int): max number of atoms in CNN model.

        Raises:
            ValueError: if adsorbate_numAtoms greater than current.

        Returns:
            dict: DOS dict with adsorbate DOS appended.

        """
        # Load adsorbate DOS file in shape (numAtoms, NEDOS, numChannels)
        adsorbate_dos = np.load(adsorbate_dosfile)


        # Zero-pad adsorbate DOS to numAtoms
        assert isinstance(adsorbate_numAtoms, int) and adsorbate_numAtoms >= 1
        if adsorbate_dos.shape[0] < adsorbate_numAtoms:
            adsorbate_dos = np.pad(adsorbate_dos, ((0, adsorbate_numAtoms - adsorbate_dos.shape[9]), (0, 0), (0, 0)))


        elif adsorbate_dos.shape[0] > adsorbate_numAtoms:
            raise ValueError(f"Current adsorbate DOS numAtoms {adsorbate_dos.shape[0]} greater than desired {adsorbate_numAtoms}.")


        # Transform adsorbate shape from (numAtoms, NEDOS, numChannels) to (NEDOS, numChannels, numAtoms)
        adsorbate_dos = np.transpose(adsorbate_dos, (1, 2, 0))


        # Append adsorbate DOS
        dos_with_adsorbate_dos = {}
        for name, dos_array in self.loaded_dos.items():
            # Expand DOS from (NEDOS, numChannels) to (NEDOS, numChannels, 1)
            dos_array = np.expand_dims(dos_array, axis=2)

            # Append adsorbate DOS array
            dos_with_adsorbate_dos[name] = np.concatenate((dos_array, adsorbate_dos), axis=2)

        return dos_with_adsorbate_dos


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

        self.loaded_dos = loaded_dos


    def load_label(self, label_csvfile, eads_col_name="adsorption_energy"):
        """Load labels from csv file.

        Args:
            label_csvfile (Path): label csv file.
            eads_col_name (str, optional): name of adsorption energy column. Defaults to "adsorption_energy".

        Returns:
            dict: label dict

        """
        # Import label csv file
        assert label_csvfile.exists()
        label_file = pd.read_csv(label_csvfile)


        # Parse labels
        labels = dict(zip(label_file["project_name"], label_file["adsorption_energy"]))

        return {str(k): float(v) for k, v in labels.items()}


# Test area
if __name__ == "__main__":
    # Manually set project name
    project = "1-doping"


    # Test loading DOS and append adsorbate DOS
    from pathlib import Path
    loader = dosLoader(
        dos_path=Path("../../1-perturbation-analysis") / "data" / project / "1-perturbed",
        dos_filename="dos_up_71.npy",
        append_adsorbate=True,
        adsorbate_dosfile=Path("../../../0-dataset/feature_DOS/adsorbate-DOS/1-CO2/dos_up_adsorbate.npy"),
        adsorbate_numAtoms=5,
        )


    # Test loading labels
    labels = loader.load_label(label_csvfile=Path(f"../data/{project}.csv"))
    print(labels)
