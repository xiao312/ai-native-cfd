from xml.etree import ElementTree

import pytest

from aicfd import AdaptiveTree, Obstacle2D, snap_to_obstacle
from aicfd.visualization import (
    CellColorMode,
    SvgOptions,
    render_geometry_svg,
    write_geometry_svg,
)

_SVG_NAMESPACE = {"svg": "http://www.w3.org/2000/svg"}


def _uniform_quadtree(level: int = 2) -> AdaptiveTree:
    tree = AdaptiveTree(dimension=2)
    for _ in range(level):
        tree.refine_many(tree.leaves)
    return tree


def _snapped_circle():
    tree = _uniform_quadtree()
    obstacle = Obstacle2D.circle((0.5, 0.5), 0.28, segments=48)
    return tree, obstacle, snap_to_obstacle(tree, obstacle)


def test_svg_contains_one_inspectable_group_per_leaf() -> None:
    tree, obstacle, snapped = _snapped_circle()

    root = ElementTree.fromstring(
        render_geometry_svg(
            tree,
            obstacle=obstacle,
            snapped=snapped,
            options=SvgOptions(color_by=CellColorMode.CLASSIFICATION),
        )
    )

    cell_groups = root.findall(".//svg:g[@class='cell']", _SVG_NAMESPACE)
    assert len(cell_groups) == len(tree)
    assert {group.attrib["data-cell-id"] for group in cell_groups} == {
        cell.stable_id for cell in tree.leaves
    }
    assert {group.attrib["data-classification"] for group in cell_groups} >= {
        "fluid",
        "cut",
    }
    assert root.find(".//svg:g[@class='obstacle']", _SVG_NAMESPACE) is not None
    assert (
        root.find(".//svg:g[@class='cut-boundaries']", _SVG_NAMESPACE) is not None
    )


def test_svg_can_show_snapped_points_and_boundary_normals() -> None:
    tree, obstacle, snapped = _snapped_circle()

    root = ElementTree.fromstring(
        render_geometry_svg(
            tree,
            obstacle=obstacle,
            snapped=snapped,
            options=SvgOptions(show_snapped_points=True, show_normals=True),
        )
    )

    points = root.findall(
        ".//svg:g[@class='snapped-points']/svg:circle", _SVG_NAMESPACE
    )
    normals = root.findall(
        ".//svg:g[@class='boundary-normals']/svg:line", _SVG_NAMESPACE
    )
    assert points
    assert len(normals) == len(snapped.cut_cells)


def test_level_coloring_does_not_require_embedded_boundary_data() -> None:
    tree = _uniform_quadtree(level=1)

    root = ElementTree.fromstring(render_geometry_svg(tree))

    assert len(root.findall(".//svg:g[@class='cell']", _SVG_NAMESPACE)) == len(tree)


def test_geometry_coloring_requires_a_matching_snapped_snapshot() -> None:
    tree = _uniform_quadtree(level=1)

    with pytest.raises(ValueError, match="requires snapped geometry"):
        render_geometry_svg(
            tree,
            options=SvgOptions(color_by=CellColorMode.FLUID_FRACTION),
        )

    obstacle = Obstacle2D.circle((0.5, 0.5), 0.2)
    snapped = snap_to_obstacle(tree, obstacle)
    tree.refine(tree.leaves[0])

    with pytest.raises(ValueError, match="stale"):
        render_geometry_svg(tree, obstacle=obstacle, snapped=snapped)


def test_svg_renderer_rejects_non_2d_trees() -> None:
    with pytest.raises(ValueError, match="2D tree"):
        render_geometry_svg(AdaptiveTree(dimension=3))


def test_write_geometry_svg_returns_and_writes_requested_path(tmp_path) -> None:
    output = tmp_path / "mesh.svg"

    returned = write_geometry_svg(output, _uniform_quadtree(level=1))

    assert returned == output
    assert output.read_text(encoding="utf-8").startswith("<?xml")
