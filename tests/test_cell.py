from fractions import Fraction

import pytest

from aicfd import Cell


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_cell_has_two_to_the_dimension_children(dimension: int) -> None:
    root = Cell(level=0, index=(0,) * dimension)

    children = root.children()

    assert len(children) == 2**dimension
    assert all(child.parent == root for child in children)
    assert all(child.level == 1 for child in children)


def test_cell_geometry_and_morton_code() -> None:
    cell = Cell(level=2, index=(2, 1))

    assert cell.normalized_bounds == ((0.5, 0.75), (0.25, 0.5))
    assert cell.normalized_measure == Fraction(1, 16)
    assert cell.morton_code == 6
    assert cell.stable_id == "2D:L2:M6"


def test_ancestor_relationship() -> None:
    ancestor = Cell(level=1, index=(1, 0, 1))
    descendant = Cell(level=3, index=(5, 1, 6))

    assert ancestor.is_ancestor_of(descendant)
    assert not descendant.is_ancestor_of(ancestor)


@pytest.mark.parametrize(
    ("level", "index", "error"),
    [
        (-1, (0,), ValueError),
        (0, (), ValueError),
        (0, (0, 0, 0, 0), ValueError),
        (1, (2,), ValueError),
        (1, (0.5,), TypeError),
    ],
)
def test_invalid_cells_are_rejected(
    level: int, index: tuple[object, ...], error: type[Exception]
) -> None:
    with pytest.raises(error):
        Cell(level=level, index=index)  # type: ignore[arg-type]
