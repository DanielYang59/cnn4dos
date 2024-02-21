"""Pytest unit test for util-convert_atom_selection."""

import pytest

from cnn4dos.utils import convert_atom_selection as cas

# TODO: move global var definition inside test class
atom_list = ["Fe", "Fe", "Co", "Co", "Ni", "Ni", "Fe", "Fe"]


class Test_convert_atom_selection:

    @pytest.mark.parametrize(
        "selections, expected",
        [
            ("Fe", [0, 1, 6, 7]),
            (["Fe", ], [0, 1, 6, 7]),
            (["Fe", "Co"], [0, 1, 2, 3, 6, 7]),
        ],
    )
    def test_convert_ele_name(self, selections, expected):
        indexes = cas(
            atom_list=atom_list,
            selections=selections,
            )

        assert indexes == expected

    @pytest.mark.parametrize(
        "selections, expected",
        [
            (1, [0, ]),
            ([1, ], [0, ]),
            ([1, 2], [0, 1]),
        ],
    )
    def test_convert_index(self, selections, expected):
        indexes = cas(
            atom_list=atom_list,
            selections=selections,
            )

        assert indexes == expected

    def test_convert_index_range(self):
        indexes = cas(
            atom_list=atom_list,
            selections="1-3",
            )

        assert indexes == [0, 1, 2]

    def test_convert_mixing(self):
        indexes = cas(
            atom_list=atom_list,
            selections=["Fe", 3, "4-6"],
            )

        assert indexes == list(range(8))

    @pytest.mark.parametrize(
        "index_start, expected",
        [
            (0, [1, 2]),
            (1, [0, 1]),
        ],
    )
    def test_index_start(self, index_start, expected):
        indexes = cas(
            atom_list=atom_list,
            selections=[1, 2],
            sel_index_start=index_start
            )

        assert indexes == expected

    @pytest.mark.parametrize(
        "sort, expected",
        [
            (True, [2, 3, 4, 5]),
            (False, [4, 5, 2, 3]),
        ],
    )
    def test_sort(self, sort, expected):
        indexes = cas(
            atom_list=atom_list,
            selections=["Ni", "Co"],
            sort=sort
            )

        assert indexes == expected

    @pytest.mark.parametrize(
        "allow_dup, expected",
        [
            (True, [2, 3, 2, 3]),
            (False, [2, 3]),
        ],
    )
    def test_duplicate(self, allow_dup, expected):
        with pytest.warns(UserWarning, match="Duplicate found in atom indexes."):
            indexes = cas(
                atom_list=atom_list,
                selections=["Co", "3-4"],
                sort=False,
                allow_duplicate=allow_dup
            )

        assert indexes == expected
