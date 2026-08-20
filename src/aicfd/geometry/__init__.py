"""Two-dimensional obstacle geometry and embedded-boundary utilities."""

from aicfd.geometry.adaptation import (
    AdaptationReport,
    DistanceBand,
    GeometryRefinementPolicy,
    adapt_to_obstacle,
)
from aicfd.geometry.embedded_boundary import (
    CellClassification,
    SnappedCell2D,
    SnappedGeometry2D,
    snap_to_obstacle,
)
from aicfd.geometry.obstacle import (
    BoundarySample2D,
    BoundaryVertex2D,
    Obstacle2D,
)

__all__ = [
    "AdaptationReport",
    "BoundarySample2D",
    "BoundaryVertex2D",
    "CellClassification",
    "DistanceBand",
    "GeometryRefinementPolicy",
    "Obstacle2D",
    "SnappedCell2D",
    "SnappedGeometry2D",
    "adapt_to_obstacle",
    "snap_to_obstacle",
]
