from math import isclose, pi

import pytest

from aicfd import Obstacle2D


def test_circle_has_expected_area_curvature_and_outward_normal() -> None:
    obstacle = Obstacle2D.circle((0.5, 0.5), 0.25, segments=128)

    maximum_area_error = 2.0e-4
    assert abs(obstacle.polygon.area - pi * 0.25**2) < maximum_area_error
    assert all(
        isclose(feature.curvature, 4.0, rel_tol=1.0e-3)
        for feature in obstacle.vertex_features
    )

    sample = obstacle.nearest_boundary((0.9, 0.5))
    assert sample.signed_distance > 0.0
    assert sample.point == pytest.approx((0.75, 0.5))
    assert sample.normal == pytest.approx((1.0, 0.0), abs=0.03)


def test_naca0012_is_a_symmetric_closed_obstacle() -> None:
    obstacle = Obstacle2D.naca4("0012", points_per_surface=81)

    lower_x, lower_y, upper_x, upper_y = obstacle.polygon.bounds
    assert (lower_x, upper_x) == pytest.approx((0.0, 1.0))
    assert lower_y == pytest.approx(-upper_y)
    assert obstacle.polygon.area > 0.08
    assert obstacle.polygon.is_valid


def test_invalid_obstacles_are_rejected() -> None:
    with pytest.raises(ValueError, match="invalid"):
        Obstacle2D([(0, 0), (1, 1), (0, 1), (1, 0)], name="bow tie")
    with pytest.raises(ValueError, match="four-digit"):
        Obstacle2D.naca4("12")
    with pytest.raises(ValueError, match="zero-thickness"):
        Obstacle2D.naca4("0000")
