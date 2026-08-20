"""Build and visualize a tiny geometry-driven quadtree without running CFD."""

from __future__ import annotations

import argparse
from pathlib import Path

from aicfd import (
    AdaptiveTree,
    DistanceBand,
    GeometryRefinementPolicy,
    Obstacle2D,
    adapt_to_obstacle,
    snap_to_obstacle,
)
from aicfd.visualization import CellColorMode, SvgOptions, write_geometry_svg


def make_problem(shape: str) -> tuple[AdaptiveTree, Obstacle2D]:
    """Return the domain and obstacle for one classic geometry check."""

    if shape == "circle":
        tree = AdaptiveTree(dimension=2, origin=(-1.0, -0.5), extent=(2.0, 1.0))
        obstacle = Obstacle2D.circle((0.0, 0.0), 0.2, segments=96)
    else:
        tree = AdaptiveTree(dimension=2, origin=(-0.5, -0.5), extent=(2.0, 1.0))
        obstacle = Obstacle2D.naca4("0012", leading_edge=(0.0, 0.0))
    return tree, obstacle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=("circle", "naca0012"), default="circle")
    parser.add_argument("--output", type=Path, default=Path("geometry-amr.svg"))
    parser.add_argument(
        "--color-by",
        choices=tuple(mode.value for mode in CellColorMode),
        default=CellColorMode.CLASSIFICATION.value,
    )
    parser.add_argument("--show-points", action="store_true")
    parser.add_argument("--show-normals", action="store_true")
    arguments = parser.parse_args()

    tree, obstacle = make_problem(arguments.shape)
    policy = GeometryRefinementPolicy(
        min_level=1,
        boundary_level=4,
        max_level=6,
        distance_bands=(DistanceBand(0.04, 4), DistanceBand(0.12, 3)),
        max_chord_error=0.0015,
        feature_angle_degrees=35.0,
        feature_level=6,
        feature_distance=0.02,
    )
    report = adapt_to_obstacle(tree, obstacle, policy)
    snapped = snap_to_obstacle(tree, obstacle)
    output = write_geometry_svg(
        arguments.output,
        tree,
        obstacle=obstacle,
        snapped=snapped,
        options=SvgOptions(
            color_by=CellColorMode(arguments.color_by),
            show_snapped_points=arguments.show_points,
            show_normals=arguments.show_normals,
        ),
    )

    print(f"shape: {obstacle.name}")
    print(f"leaves: {report.leaves_before} -> {report.leaves_after}")
    print(f"level range: {min(cell.level for cell in tree.leaves)}..{tree.max_level}")
    print(f"cut cells: {len(snapped.cut_cells)}")
    print(f"fluid area: {snapped.fluid_area:.8f}")
    print(f"preview: {output.resolve()}")


if __name__ == "__main__":
    main()
