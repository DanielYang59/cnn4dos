"""Pytest unit test for util-plot_edos."""

from unittest.mock import patch

import numpy as np
import pytest

from cnn4dos.plotters import plot_edos_1d


class Test_plot_edos_1d:
    test_energy_arr = np.linspace(0, 10, 100)
    test_edos_arr = np.sin(test_energy_arr)

    @patch("matplotlib.pyplot.figure")
    def test_plot_edos_1d(self, mock_fig) -> None:
        plot_edos_1d(self.test_edos_arr, self.test_energy_arr)

        mock_fig.assert_called()

    @pytest.mark.parametrize(
        "invalid_edos",
        [
            [
                0,
                1,
            ],
            np.random.rand(2, 2),
        ],
    )
    def test_invalid_edos(self, invalid_edos) -> None:
        with pytest.raises(TypeError, match="Expect 1D eDOS array."):
            plot_edos_1d(invalid_edos)

    def test_edos_energy_shape_diff(self) -> None:
        with pytest.raises(ValueError, match="eDOS and energy shape differ."):
            plot_edos_1d(self.test_edos_arr, np.append(self.test_edos_arr, 0))
