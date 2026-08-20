"""Internal helpers that map adaptive cells to Shapely geometry."""

from shapely.geometry import Polygon, box

from aicfd.representation import AdaptiveTree, Cell


def cell_polygon_2d(tree: AdaptiveTree, cell: Cell) -> Polygon:
    """Return the rectangular polygon occupied by one 2D tree cell."""

    if tree.dimension != 2:
        raise ValueError("obstacle geometry currently requires a 2D tree")
    (lower_x, upper_x), (lower_y, upper_y) = tree.physical_bounds(cell)
    return box(lower_x, lower_y, upper_x, upper_y)
