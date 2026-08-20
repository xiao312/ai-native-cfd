from aicfd import (
    AdaptiveTree,
    Cell,
    DistanceBand,
    GeometryRefinementPolicy,
    Obstacle2D,
    adapt_to_obstacle,
)


def test_distance_bands_set_local_target_levels() -> None:
    tree = AdaptiveTree(dimension=2)
    obstacle = Obstacle2D.circle((0.5, 0.5), 0.15, segments=64)
    policy = GeometryRefinementPolicy(
        min_level=1,
        boundary_level=4,
        max_level=5,
        distance_bands=(DistanceBand(0.05, 4), DistanceBand(0.2, 3)),
        feature_angle_degrees=None,
    )

    near = Cell(level=4, index=(5, 7))
    far = Cell(level=4, index=(0, 0))

    assert policy.target_level(tree, near, obstacle) == 4
    assert policy.target_level(tree, far, obstacle) == 1


def test_sharp_corner_gets_more_resolution_than_a_flat_side() -> None:
    tree = AdaptiveTree(dimension=2)
    obstacle = Obstacle2D(
        [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)],
        name="square",
    )
    policy = GeometryRefinementPolicy(
        min_level=1,
        boundary_level=3,
        max_level=6,
        feature_angle_degrees=45.0,
        feature_level=6,
        feature_distance=0.0,
    )

    corner_cell = Cell(level=4, index=(3, 3))
    flat_side_cell = Cell(level=4, index=(8, 3))

    assert policy.target_level(tree, corner_cell, obstacle) == 6
    assert policy.target_level(tree, flat_side_cell, obstacle) == 3


def test_curvature_rule_uses_a_chord_error_target() -> None:
    tree = AdaptiveTree(dimension=2, origin=(-1.0, -0.5), extent=(2.0, 1.0))
    obstacle = Obstacle2D.circle((0.0, 0.0), 0.25, segments=64)
    policy = GeometryRefinementPolicy(
        min_level=0,
        boundary_level=0,
        max_level=7,
        max_chord_error=0.00125,
        feature_angle_degrees=None,
    )

    # kappa ~= 4 and e ~= kappa*h^2/8 gives h <= 0.05. The longest
    # level-6 cell side is 2/64 = 0.03125, while level 5 is too large.
    assert policy.target_level(tree, tree.root, obstacle) == 6


def test_adaptation_refines_new_geometry_and_coarsens_old_geometry() -> None:
    tree = AdaptiveTree(dimension=2)
    first_obstacle = Obstacle2D.circle((0.3, 0.5), 0.12, segments=48)
    moved_obstacle = Obstacle2D.circle((0.7, 0.5), 0.12, segments=48)
    policy = GeometryRefinementPolicy(
        min_level=1,
        boundary_level=4,
        max_level=4,
        distance_bands=(DistanceBand(0.08, 3),),
        feature_angle_degrees=None,
        coarsening_hysteresis=0.1,
    )

    adapt_to_obstacle(tree, first_obstacle, policy)
    moved_report = adapt_to_obstacle(tree, moved_obstacle, policy)

    assert moved_report.refinement_count > 0
    assert moved_report.coarsening_count > 0
    assert tree.locate((0.58, 0.5)).level == 4
    assert tree.locate((0.18, 0.5)).level < 4
    assert tree.is_balanced()
    tree.assert_valid()
