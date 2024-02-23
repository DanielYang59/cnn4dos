"""Pack eDOS as deep learning dataset.

# NOTE:
# Need to be thoughtful not to direlctly load the file to memory,
# but use TF to load them on the fly to reduce memory footage.
"""

from typing import Any

from cnn4dos.data import Edos


class Sample:
    """
    Represents a sample consisting of eDOS data, a float label,
        and optional metadata.

    Attributes:
        feature (Edos): The eDOS data for the sample.
        label (float): The float label associated with the sample.
        metadata (Any, optional): Additional metadata associated.

    Raises:
        TypeError: If feature is not of type Edos.
        ValueError: If label is not a float.
    """

    def __init__(
        self, feature: Edos, label: float, metadata: Any = None
    ) -> None:
        # Check args
        if not isinstance(feature, Edos):
            raise TypeError("Expect feature as Edos type.")

        if not isinstance(label, float):
            raise ValueError("Expect label as float.")

        self.feature = feature
        self.label = label

        self.metadata = metadata


class Dataset:
    """
    Collection of Samples.

    Attributes:
        samples (list[Sample]): A list containing the samples.
    """

    def __init__(self) -> None:
        self.samples: list[Sample] = []

    def add_sample(self, sample: Sample) -> None:
        # Check arg: sample
        if not isinstance(sample, Sample):
            raise TypeError("Expect sample in type Sample.")

        self.samples.append(sample)
