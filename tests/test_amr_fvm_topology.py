import numpy as np
import pytest

from aicfd import (
    AdaptiveTree,
    AMRHierarchy,
    Cell,
    FaceField,
    FaceRelation,
    FaceTopology,
    FieldLocation,
    FieldSpec,
    GhostSourceKind,
    GhostTopology,
    TreeLayout,
    ValueRepresentation,
    flux_divergence,
)


def _uniform_level_one(dimension: int) -> tuple[TreeLayout, FaceTopology]:
    tree = AdaptiveTree(dimension)
    tree.refine(tree.root)
    layout = TreeLayout.from_tree(tree)
    return layout, FaceTopology(layout)


@pytest.mark.parametrize(
    ("dimension", "face_count", "boundary_count"),
    [(1, 3, 2), (2, 12, 8), (3, 36, 24)],
)
def test_uniform_face_topology_counts_and_constant_flux_divergence(
    dimension: int,
    face_count: int,
    boundary_count: int,
) -> None:
    layout, topology = _uniform_level_one(dimension)
    normal_x_flux = [face.normal[0] for face in topology.faces]
    field = FaceField(
        FieldSpec(
            "x_flux",
            location=FieldLocation.FACE,
            value_representation=ValueRepresentation.FACE_AVERAGE,
        ),
        topology,
        normal_x_flux,
    )

    assert len(topology) == face_count
    assert len(topology.boundary_faces) == boundary_count
    assert len(topology.interior_faces) == face_count - boundary_count
    assert len(topology.key_to_index) == face_count
    assert all(len(topology.incidences(cell)) == 2 * dimension for cell in layout.cells)
    np.testing.assert_allclose(flux_divergence(field), 0.0, atol=1.0e-14)


def test_hierarchy_distinguishes_active_leaves_from_covered_parents() -> None:
    tree = AdaptiveTree(2)
    tree.refine(tree.root)
    refined_parent = Cell(1, (0, 0))
    children = tree.refine(refined_parent)
    layout = TreeLayout.from_tree(tree)

    hierarchy = AMRHierarchy(layout)

    assert len(hierarchy.nodes) == 9
    assert len(hierarchy.active_cells) == 7
    assert hierarchy.refined_cells == (tree.root, refined_parent)
    assert hierarchy.at_level(0).refined_cells == (tree.root,)
    assert hierarchy.at_level(1).active_cells == (
        Cell(1, (0, 1)),
        Cell(1, (1, 0)),
        Cell(1, (1, 1)),
    )
    assert hierarchy.at_level(1).refined_cells == (refined_parent,)
    assert hierarchy.at_level(2).active_cells == children
    assert hierarchy.children_of(refined_parent) == children
    assert hierarchy.parent_of(children[0]) == refined_parent
    assert not hierarchy.is_active(refined_parent)
    assert hierarchy.is_active(children[0])
    assert len(hierarchy.hierarchy_edges) == 8


def test_coarse_face_is_partitioned_into_fine_sized_segments() -> None:
    tree = AdaptiveTree(2)
    tree.refine(tree.root)
    tree.refine(Cell(1, (0, 0)))
    topology = FaceTopology(TreeLayout.from_tree(tree))
    coarse_right = Cell(1, (1, 0))

    left_side = topology.cell_side_faces(coarse_right, axis=0, side=-1)

    assert len(topology.faces) == 20
    assert len(topology.coarse_fine_faces) == 4
    assert len(left_side) == 2
    assert all(face.relation is FaceRelation.COARSE_FINE for face in left_side)
    assert all(face.coarse_cell == coarse_right for face in left_side)
    assert {face.fine_cell for face in left_side} == {
        Cell(2, (1, 0)),
        Cell(2, (1, 1)),
    }
    assert sum(face.area for face in left_side) == pytest.approx(0.5)
    assert all(face.area == pytest.approx(0.25) for face in left_side)


def test_ghost_topology_records_each_source_recipe() -> None:
    tree = AdaptiveTree(2)
    tree.refine(tree.root)
    tree.refine(Cell(1, (0, 0)))
    topology = FaceTopology(TreeLayout.from_tree(tree))

    ghosts = GhostTopology(topology)

    assert len(ghosts.slots) == 7 * 2 * 2
    restriction = ghosts.at(Cell(1, (1, 0)), axis=0, side=-1)
    assert restriction.source_kind is GhostSourceKind.FINE_RESTRICTION
    assert restriction.source_cells == (
        Cell(2, (1, 0)),
        Cell(2, (1, 1)),
    )

    prolongation = ghosts.at(Cell(2, (1, 0)), axis=0, side=1)
    assert prolongation.source_kind is GhostSourceKind.COARSE_PROLONGATION
    assert prolongation.source_cells == (Cell(1, (1, 0)),)

    same_level = ghosts.at(Cell(2, (0, 0)), axis=0, side=1)
    assert same_level.source_kind is GhostSourceKind.SAME_LEVEL
    assert same_level.source_cells == (Cell(2, (1, 0)),)

    boundary = ghosts.at(Cell(2, (0, 0)), axis=1, side=-1)
    assert boundary.source_kind is GhostSourceKind.PHYSICAL_BOUNDARY
    assert boundary.source_cells == ()
    assert boundary.boundary_label == "y-"


def test_oriented_interior_flux_has_equal_and_opposite_cell_contributions() -> None:
    layout, topology = _uniform_level_one(1)
    values = np.zeros(len(topology))
    interior = topology.interior_faces[0]
    values[topology.index(interior)] = 2.0
    flux = FaceField(
        FieldSpec(
            "flux",
            location=FieldLocation.FACE,
            value_representation=ValueRepresentation.FACE_AVERAGE,
        ),
        topology,
        values,
    )

    divergence = flux_divergence(flux)

    np.testing.assert_allclose(divergence, [4.0, -4.0])
    volumes = np.array([layout.cell_measure(cell) for cell in layout.cells])
    assert float(np.sum(divergence * volumes)) == pytest.approx(0.0)


def test_face_field_rejects_cell_metadata_and_wrong_shape() -> None:
    _, topology = _uniform_level_one(1)

    with pytest.raises(ValueError, match="FieldLocation.FACE"):
        FaceField(FieldSpec("bad"), topology, np.zeros(len(topology)))
    with pytest.raises(ValueError, match="shape"):
        FaceField(
            FieldSpec(
                "bad",
                location=FieldLocation.FACE,
                value_representation=ValueRepresentation.FACE_INTEGRAL,
            ),
            topology,
            np.zeros(len(topology) + 1),
        )
