import pytest
from shapely.geometry import Point

from aicfd import (
    AdaptiveTree,
    CellClassification,
    DistanceBand,
    GeometryRefinementPolicy,
    Obstacle2D,
    adapt_to_obstacle,
    snap_to_obstacle,
)


def test_snapping_classifies_cells_and_conserves_fluid_area() -> None:
    tree = AdaptiveTree(dimension=2)
    obstacle = Obstacle2D.circle((0.5, 0.5), 0.2, segments=96)
    policy = GeometryRefinementPolicy(
        min_level=1,
        boundary_level=5,
        max_level=5,
        distance_bands=(DistanceBand(0.08, 4),),
        feature_angle_degrees=None,
    )
    adapt_to_obstacle(tree, obstacle, policy)

    snapped = snap_to_obstacle(tree, obstacle)

    assert snapped.cut_cells
    assert snapped.fluid_cells
    assert snapped.solid_cells
    assert snapped.fluid_area == pytest.approx(1.0 - obstacle.polygon.area)
    assert all(0.0 < cell.fluid_fraction < 1.0 for cell in snapped.cut_cells)


def test_snapped_points_lie_on_the_input_polyline() -> None:
    tree = AdaptiveTree(dimension=2)
    obstacle = Obstacle2D(
        [(0.2, 0.25), (0.8, 0.25), (0.65, 0.75), (0.35, 0.75)],
        name="trapezoid",
    )
    tree.refine(tree.root)
    tree.refine_many(tree.leaves)

    snapped = snap_to_obstacle(tree, obstacle)
    points = tuple(
        point for cell in snapped.cut_cells for point in cell.snapped_boundary_points
    )

    assert points
    assert all(obstacle.boundary.distance(Point(point)) < 1.0e-12 for point in points)
    assert all(
        cell.classification is CellClassification.CUT for cell in snapped.cut_cells
    )


def test_snapping_does_not_move_or_replace_tree_cells() -> None:
    tree = AdaptiveTree(dimension=2)
    tree.refine(tree.root)
    leaves_before = tree.leaves

    snapped = snap_to_obstacle(tree, Obstacle2D.circle((0.5, 0.5), 0.2))

    assert tree.leaves == leaves_before
    assert tuple(cell.cell for cell in snapped.cells) == leaves_before
