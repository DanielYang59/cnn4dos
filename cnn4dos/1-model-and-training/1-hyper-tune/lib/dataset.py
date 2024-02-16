"""Load and manipulate eDOS dataset for CNN."""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


class Dataset:
    """Dataset class for loading and manipulating eDOS dataset.

    Attributes:
        feature (dict): eDOS feature,
            key is "{substrate}{keysep}{adsorbate}{keysep}is/fs",
            value is eDOS array
        numFeature (int): total number of samples
        featureKeySep (str): separator used in dict keys
        substrates (list):
        adsorbates (list):

    """

    def __init__(self) -> None:
        pass

    def load_feature(
        self,
        path,
        substrates,
        adsorbates,
        centre_atoms,
        states=("is", "fs"),
        spin="up",
        load_augment=False,
        augmentations=None,
        keysep=":",
        remove_ghost=False,
    ) -> None:
        """Load eDOS dataset feature from given list of dirs.

        Args:
            path (Path): path to dataset dir
            substrates (list): list of substrates to load
            adsorbates (list): list of adsorbates to load
            centre_atoms (dict): centre atom index dict (index starts from 1)
            filename (str): name of the eDOS file under each dir
            keysep (str): separator for dir and project name in dataset dict
            states (tuple): list of states, "is" for initial state,
                "fs" for final state
            spin (str): load spin "up" or "down" DOS, or "both"
            load_augment (bool): load augmentation data or not,
                augmented substrate should end with "_aug"
            augmentations (list): list of augmentation distances
            remove_ghost (bool): remove ghost state (first point of NEDOS)

        Notes:
            1. eDOS array in (NEDOS, orbital) shape
            2. feature dict key is
                "{substrate}{keysep}{adsorbate}{keysep}is/fs"
                (is for initial state, fs for final state)
            3. Spin up eDOS should be named "dos_up.npy", down "dos_down.npy"

        """
        # Check args
        assert path.is_dir()
        assert isinstance(substrates, list)
        assert isinstance(adsorbates, list)
        assert isinstance(centre_atoms, dict)
        for index in centre_atoms.values():
            assert isinstance(index, int) and index >= 1
        for state in states:
            assert state in {"is", "fs"}
        assert spin in {"up", "down", "both"}
        assert isinstance(load_augment, bool)
        assert isinstance(remove_ghost, bool)

        # Append augmentation to substrates if required
        if load_augment:
            assert isinstance(augmentations, list)
            for i in augmentations:
                assert isinstance(i, str)
            substrates.extend([f"{i}_aug" for i in substrates])
            print(f"Augmentation data would be loaded: {augmentations}")

        # Warning user if ghost removal activated
        if remove_ghost:
            warnings.warn("Ghost state removal activated.")

        # Update attrib
        self.substrates = substrates
        self.adsorbates = adsorbates

        # Import eDOS as numpy array
        feature_data = {}
        for sub in substrates:
            # Get centre atom index from dict
            centre_atom_index = centre_atoms[sub.replace("_aug", "")]

            for ads in adsorbates:
                for state in states:
                    # Compile path
                    directory = os.path.join(path, sub, f"{ads}_{state}")
                    assert os.path.isdir(directory)

                    # Loop through all directories to load DOS
                    for folder in os.listdir(directory):
                        if os.path.isdir(os.path.join(directory, folder)) and (
                            os.path.exists(
                                os.path.join(
                                    directory, folder,
                                    f"dos_up_{centre_atom_index}.npy"
                                )
                            )
                            or os.path.exists(
                                os.path.join(
                                    directory,
                                    folder,
                                    f"dos_down_{centre_atom_index}.npy",
                                )
                            )
                        ):
                            # Do augmentation distance check for augmented data
                            if (
                                sub.endswith("_aug")
                                and folder.split("_")[-1] not in augmentations
                            ):
                                continue

                            # Compile dict key as:
                            # "{substrate}{keysep}{adsorbate}{keysep}{state}"
                            key = f"{sub}{keysep}{ads}{keysep}{state}{keysep}{folder}"

                            # Parse data as numpy array into dict
                            # Load spin up
                            if spin == "up":
                                arr = np.load(
                                    os.path.join(
                                        directory,
                                        folder,
                                        f"dos_up_{centre_atom_index}.npy",
                                    )
                                )
                            elif spin == "down":
                                arr = np.load(
                                    os.path.join(
                                        directory,
                                        folder,
                                        f"dos_down_{centre_atom_index}.npy",
                                    )
                                )
                            else:  # load both spin-up and down
                                arr_up = np.load(
                                    os.path.join(
                                        directory,
                                        folder,
                                        f"dos_up_{centre_atom_index}.npy",
                                    )
                                )  # (NEDOS, numOrbital)
                                arr_down = np.load(
                                    os.path.join(
                                        directory,
                                        folder,
                                        f"dos_down_{centre_atom_index}.npy",
                                    )
                                )  # (NEDOS, numOrbital)
                                arr = np.stack(
                                    [arr_up, arr_down], axis=2
                                )  # (NEDOS, numOrbital, 2)

                            # Remove "ghost state": zero out first point
                            if remove_ghost:
                                arr[0] = 0.0

                            # Update dict value
                            # shape (NEDOS, numOrbital)
                            feature_data[key] = arr

        # Update attrib
        self.feature = feature_data
        self.numFeature = len(feature_data)
        self.featureKeySep = keysep

    def scale_feature(self, mode) -> None:
        """Scale feature arrays.

        Args:
            mode (str): scaling mode

        Notes:
            arr shape: (NEDOS, orbital)

        """
        # Check args
        assert mode in {"normalization", "none", "standardization"}

        # Skip "none" mode
        if mode != "none":
            # Loop through dataset and perform scaling
            for key, arr in self.feature.items():
                # Perform scaling for each channel
                # expect shape in (NEDOS, numOrbitals, numChannels)
                assert len(arr.shape) == 3
                scaled_arr = []

                for channel_index in range(arr.shape[2]):
                    if mode == "normalization":
                        channel = normalize(
                            arr[:, :, channel_index], axis=0, norm="max"
                        )
                    elif mode == "standardization":
                        raise RuntimeError("Still working on.")

                    scaled_arr.append(channel)

                # Update dataset
                scaled_arr = np.stack(scaled_arr, axis=2)
                self.feature[key] = scaled_arr

    def load_label(self, label_dir) -> None:
        """Load labels based on names of feature files.

        Args:
            label_dir (str): label csv files directory.

        """
        # Check args
        assert os.path.isdir(label_dir)

        # Load label csv as pd DataFrame
        labels_source = {}
        for file in os.listdir(label_dir):
            if file.endswith(".csv") and not file.startswith("."):
                labels_source[file.replace(".csv", "")] = pd.read_csv(
                    os.path.join(label_dir, file)
                )

        # Fetch label from source dict
        labels = {}
        for (
            key
            # key format "{substrate}{keysep}{adsorbate}{keysep}{state}"
        ) in self.feature:
            # Unpack key
            substrate, adsorbate, state, project = key.split(self.featureKeySep)

            # Find label from dataframe
            df = labels_source[f"{substrate}_{state}"]
            df.index = df.iloc[:, 0]  # set first column as headers
            try:
                labels[key] = float(df.loc[project][adsorbate])
            except KeyError:
                raise KeyError(f'Label for key:"{key}" not found.')

        self.label = labels

    def append_adsorbate_DOS(
        self,
        adsorbate_dos_dir,
        dos_name="dos_up_adsorbate.npy"
            ) -> None:
        """Append adsorbate eDOS to metal DOS.

        Args:
            adsorbate_dos_dir (str): adsorbate eDOS directory.
            dos_name (str, optional): name of adsorbate DOS.
                Defaults to "dos_up_adsorbate.npy".

        """
        # Check args
        assert os.path.isdir(adsorbate_dos_dir)

        # Loop through dataset and append adsorbate DOS
        for key, arr in self.feature.items():
            # Get adsorbate name
            mol_name = key.split(":")[1]
            # Load adsorbate DOS
            mol_dos_arr = np.load(
                os.path.join(adsorbate_dos_dir, mol_name, dos_name)
            )

            # Append to original DOS
            arr = np.expand_dims(
                arr, axis=0
            )  # reshape original eDOS from (4000, 9) to (1, 4000, 9)
            arr = np.concatenate([arr, mol_dos_arr])

            # Swap (6, 4000, 9) to (4000, 9, 6)
            arr = np.swapaxes(arr, 0, 1)
            arr = np.swapaxes(arr, 1, 2)

            # Update feature dict
            self.feature[key] = arr
