"""Load and preprocess descriptor data for plotting."""

from pathlib import Path

import pandas as pd


class Descriptors:
    def __init__(
        self,
        adsorption_energy_file,
        element_descriptor_file,
        electronic_descriptor_file,
    ) -> None:
        # Load adsorption energy
        self.__load_adsorption_energy(adsorption_energy_file)

        # Load element descriptors
        self.__load_element_descriptor(element_descriptor_file)

        # Load electronic descriptors
        self.__load_electronic_descriptor(electronic_descriptor_file)

        # Merge all descriptors
        self.__merge_descriptors()

    def __load_adsorption_energy(self, path):
        """Load adsorption energy from csv file.

        Args:
            path (Path): _description_

        """
        assert path.exists()
        self.adsorption_energy = pd.read_csv(path)

    def __load_element_descriptor(self, path):
        """Load element (elementary) descriptors from csv file.

        Args:
            path (Path): _description_

        """
        assert path.exists()
        self.element_descriptor = pd.read_csv(path)

    def __load_electronic_descriptor(self, path):
        """Load electronic descriptors from csv file.

        Args:
            path (Path): _description_

        """
        assert path.exists()
        self.electronic_descriptor = pd.read_csv(path)

    def __merge_descriptors(self):
        """Merge adsorption energy, element descriptors
        and electronic descriptors.
        """
        # Check substrates of adsorption energy and electronic descriptors
        if not self.adsorption_energy["substrate"].equals(
            self.electronic_descriptor["substrate"]
        ):
            raise ValueError(
                (
                    "Substrate lists in adsorption energy and "
                    "electronic descriptors don't match."
                )
            )

        # Check metals of adsorption energy and electronic descriptors
        if not self.adsorption_energy["metal"].equals(
            self.electronic_descriptor["metal"]
        ):
            raise ValueError(
                (
                    "Metal lists in adsorption energy and "
                    "electronic descriptors don't match."
                )
            )

        # Duplicate element descriptors
        numSubstrates = len(set(self.adsorption_energy["substrate"]))
        dup_element_descriptor = pd.concat(
            [self.element_descriptor] * numSubstrates, ignore_index=True
        )

        # Check metal order
        if not [i.split("-")[0] for i in dup_element_descriptor["metal"]] != list(
            self.adsorption_energy["metal"]
        ):
            raise ValueError(
                (
                    "Metal lists in adsorption energy and elementary "
                    "descriptors don't match."
                )
            )

        # Merge three descriptors
        self.merged_descriptors = pd.concat(
            [
                self.adsorption_energy,
                dup_element_descriptor,
                self.electronic_descriptor,
            ],
            axis=1,
        )

        # Remove duplicated metal/substrate columns
        self.merged_descriptors = self.merged_descriptors.loc[
            :, ~self.merged_descriptors.columns.duplicated()
        ]


# Test area
if __name__ == "__main__":
    loader = Descriptors(
        adsorption_energy_file=Path("../data/EadsCO.csv"),
        element_descriptor_file=Path("../data/element-descriptors.csv"),
        electronic_descriptor_file=Path("../data/electronic-descriptors.csv"),
    )
    print(loader.merged_descriptors)
