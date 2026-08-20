"""Conservative 2D clipping of quadtree cells against an obstacle."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from shapely.geometry import LinearRing, LineString, Point
from shapely.geometry.base import BaseGeometry

from aicfd.geometry._cell import cell_polygon_2d
from aicfd.geometry.obstacle import Obstacle2D, Point2D
from aicfd.representation import AdaptiveTree, Cell


class CellClassification(str, Enum):
    """How an obstacle occupies one Cartesian cell."""

    FLUID = "fluid"
    CUT = "cut"
    SOLID = "solid"


@dataclass(frozen=True, slots=True)
class SnappedCell2D:
    """Geometry metadata for one leaf after embedded-boundary clipping."""

    cell: Cell
    classification: CellClassification
    cell_area: float
    fluid_fraction: float
    signed_distance: float
    nearest_boundary_point: Point2D
    boundary_normal: Point2D
    snapped_boundary_points: tuple[Point2D, ...]
    fluid_region: BaseGeometry = field(repr=False, compare=False)
    boundary_fragment: BaseGeometry = field(repr=False, compare=False)

    @property
    def fluid_area(self) -> float:
        """Physical fluid area represented by this leaf."""

        return self.cell_area * self.fluid_fraction


@dataclass(frozen=True, slots=True)
class SnappedGeometry2D:
    """All clipped leaves for one tree and obstacle."""

    obstacle_name: str
    cells: tuple[SnappedCell2D, ...]

    def __len__(self) -> int:
        return len(self.cells)

    @property
    def fluid_cells(self) -> tuple[SnappedCell2D, ...]:
        """Leaves containing only fluid."""

        return tuple(
            cell
            for cell in self.cells
            if cell.classification is CellClassification.FLUID
        )

    @property
    def cut_cells(self) -> tuple[SnappedCell2D, ...]:
        """Leaves intersected by the obstacle boundary."""

        return tuple(
            cell
            for cell in self.cells
            if cell.classification is CellClassification.CUT
        )

    @property
    def solid_cells(self) -> tuple[SnappedCell2D, ...]:
        """Leaves containing no fluid area."""

        return tuple(
            cell
            for cell in self.cells
            if cell.classification is CellClassification.SOLID
        )

    @property
    def fluid_area(self) -> float:
        """Total fluid area, summed conservatively over all leaves."""

        return sum(cell.fluid_area for cell in self.cells)

    def for_cell(self, cell: Cell) -> SnappedCell2D:
        """Return geometry data for a particular leaf."""

        for snapped_cell in self.cells:
            if snapped_cell.cell == cell:
                return snapped_cell
        raise KeyError(f"no snapped geometry exists for {cell}")


def _coordinates_on_boundary(geometry: BaseGeometry) -> tuple[Point2D, ...]:
    coordinates: list[Point2D] = []

    if geometry.is_empty:
        return ()
    if isinstance(geometry, Point):
        coordinates.append((float(geometry.x), float(geometry.y)))
    elif isinstance(geometry, (LineString, LinearRing)):
        coordinates.extend((float(x), float(y)) for x, y in geometry.coords)
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            coordinates.extend(_coordinates_on_boundary(part))

    # Different clipped line pieces often share an endpoint. Keep one deterministic
    # copy without rounding away meaningful geometric precision.
    return tuple(sorted(dict.fromkeys(coordinates)))


def snap_to_obstacle(
    tree: AdaptiveTree,
    obstacle: Obstacle2D,
    *,
    relative_area_tolerance: float = 1.0e-12,
) -> SnappedGeometry2D:
    """Clip every leaf against an obstacle and retain exact boundary intersections.

    No Cartesian vertex is moved. Instead, each cut leaf gets its exact fluid-area
    fraction and the piece of the input polyline that crosses it. This embedded-
    boundary interpretation is conservative and cannot invert a mesh cell.
    """

    if tree.dimension != 2:
        raise ValueError("obstacle snapping currently requires a 2D tree")
    if relative_area_tolerance < 0.0:
        raise ValueError("relative_area_tolerance must be non-negative")

    snapped_cells: list[SnappedCell2D] = []
    for cell in tree.leaves:
        cell_shape = cell_polygon_2d(tree, cell)
        cell_area = cell_shape.area
        fluid_region = cell_shape.difference(obstacle.polygon)
        fluid_area = fluid_region.area
        solid_area = max(0.0, cell_area - fluid_area)
        tolerance = relative_area_tolerance * cell_area
        if fluid_area <= tolerance:
            classification = CellClassification.SOLID
            fluid_fraction = 0.0
        elif solid_area <= tolerance:
            classification = CellClassification.FLUID
            fluid_fraction = 1.0
        else:
            classification = CellClassification.CUT
            fluid_fraction = max(0.0, min(1.0, fluid_area / cell_area))

        boundary_fragment = cell_shape.intersection(obstacle.boundary)
        boundary_sample = obstacle.nearest_boundary(tree.cell_center(cell))
        snapped_cells.append(
            SnappedCell2D(
                cell=cell,
                classification=classification,
                cell_area=cell_area,
                fluid_fraction=fluid_fraction,
                signed_distance=boundary_sample.signed_distance,
                nearest_boundary_point=boundary_sample.point,
                boundary_normal=boundary_sample.normal,
                snapped_boundary_points=_coordinates_on_boundary(boundary_fragment),
                fluid_region=fluid_region,
                boundary_fragment=boundary_fragment,
            )
        )

    return SnappedGeometry2D(obstacle_name=obstacle.name, cells=tuple(snapped_cells))
