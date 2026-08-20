import pytest

from aicfd import AdaptiveTree, Cell


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_refine_and_coarsen_in_every_dimension(dimension: int) -> None:
    tree = AdaptiveTree(dimension=dimension)

    children = tree.refine(tree.root)

    assert len(tree) == 2**dimension
    assert len(children) == 2**dimension
    assert tree.normalized_leaf_measure == 1
    tree.assert_valid()

    tree.coarsen(tree.root)

    assert tree.leaves == (tree.root,)
    tree.assert_valid()


def test_internal_nodes_and_hierarchy_edges_are_inferred() -> None:
    tree = AdaptiveTree(dimension=2)
    level_one = tree.refine(tree.root)
    tree.refine(level_one[0])

    assert len(tree.leaves) == 7
    assert len(tree.nodes) == 9
    assert len(tree.hierarchy_edges) == 8
    assert all(parent == child.parent for parent, child in tree.hierarchy_edges)


def test_physical_geometry_and_point_location() -> None:
    tree = AdaptiveTree(dimension=2, origin=(-1.0, 2.0), extent=(4.0, 2.0))
    level_one = tree.refine(tree.root)
    tree.refine(level_one[0])

    lower_left = tree.locate((-0.75, 2.25))
    upper_boundary = tree.locate((3.0, 4.0))

    assert lower_left == Cell(level=2, index=(0, 0))
    assert upper_boundary == Cell(level=1, index=(1, 1))
    assert tree.cell_size(lower_left) == (1.0, 0.5)
    assert tree.cell_measure(lower_left) == 0.5
    assert tree.cell_center(lower_left) == (-0.5, 2.25)


def test_coarse_cell_has_multiple_fine_face_neighbors() -> None:
    tree = AdaptiveTree(dimension=2)
    level_one = tree.refine(tree.root)
    tree.refine(Cell(level=1, index=(0, 0)))

    coarse_right = Cell(level=1, index=(1, 0))
    fine_left_neighbors = {
        cell
        for cell in tree.face_neighbors(coarse_right)
        if cell.level == 2 and cell.index[0] == 1
    }

    assert fine_left_neighbors == {
        Cell(level=2, index=(1, 0)),
        Cell(level=2, index=(1, 1)),
    }
    assert level_one[0] not in tree


def test_balance_refines_cells_that_are_too_coarse() -> None:
    tree = AdaptiveTree(dimension=2)
    tree.refine(tree.root)
    tree.refine(Cell(level=1, index=(0, 0)))
    tree.refine(Cell(level=2, index=(1, 0)))

    assert not tree.is_balanced()

    automatically_refined = tree.balance()

    assert Cell(level=1, index=(1, 0)) in automatically_refined
    assert tree.is_balanced()
    tree.assert_valid()


def test_serialization_round_trip() -> None:
    tree = AdaptiveTree(dimension=3, origin=(-1, -2, -3), extent=(2, 4, 6))
    children = tree.refine(tree.root)
    tree.refine(children[-1])

    restored = AdaptiveTree.from_dict(tree.to_dict())

    assert restored.dimension == tree.dimension
    assert restored.origin == tree.origin
    assert restored.extent == tree.extent
    assert restored.leaves == tree.leaves


def test_invalid_operations_are_explained() -> None:
    tree = AdaptiveTree(dimension=1)

    with pytest.raises(ValueError, match="outside"):
        tree.locate((1.1,))
    with pytest.raises(ValueError, match="non-leaf"):
        tree.refine(Cell(level=1, index=(0,)))
    with pytest.raises(ValueError, match="every child"):
        tree.coarsen(tree.root)
