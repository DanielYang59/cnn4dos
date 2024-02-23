"""Perform eDOS occlusion experiments."""

import warnings

import numpy as np

from cnn4dos.data.edos import Edos


class Occlusion:
    """eDOS occlusion experiments.

    Intro:
        Similar to the Occlusion technique in Computer Vision,
        in the context of using eDOS for deep learning, zero-patch makes
        more physical sense, representing shielding electronic states
        of a section (electron removal from corresponding energy states). I
        still don't have a solid proposal for the physical meaning behind this,
        so feel free to open a discussion.

        Meanwhile in my first attempt, I slid the mask along the energy axis
        to investigate energy-wise contribution from eDOS to label (in my case
        it was adsorption energy). But as I'm attempting to make it a general
        purpose package, you could now choose the axis with the label.
    """

    def __init__(
        self,
        edos: Edos,
        stride: int = 1,
        patch: float = 0.0,
        axis: str = "energy",
    ) -> None:
        # Check arg: edos
        if not isinstance(edos, Edos):
            raise TypeError("Expect type Edos for edos.")

        # Check arg: stride
        if not isinstance(stride, int) and stride > 0:
            raise ValueError("Expect stride as a positive integer.")
        elif stride > edos.get_shape(axis):
            raise ValueError("Stride greater than total size.")

        # Check arg: patch
        if patch != 0.0:
            warnings.warn("patch changed. Make sure it's intended.")

        # Check arg: axis
        if axis not in {"orbital", "energy", "atom", "spin"}:
            raise ValueError("Invalid axis name, atom/energy/orbital/spin.")

    def generate(self) -> np.ndarray:
        pass
