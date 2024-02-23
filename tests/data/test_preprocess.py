"""Pytest unit test for preprocess."""

import numpy as np
import pytest

from cnn4dos.data.edos import Edos
from cnn4dos.data.preprocess import remove_ghost_states


class Test_remove_ghost_states:
    def test_remove_ghost_states(self) -> None:
        edos = Edos(
            edos_arr=np.ones((3, 4, 5, 6)),
            axes=("orbital", "energy", "atom", "spin"),
        )

        new_edos = remove_ghost_states(edos, width=1)

        assert np.all(new_edos.edos_arr[:, :1, :, :] == 0.0)

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

            remove_ghost_states(edos, invalid_width)
