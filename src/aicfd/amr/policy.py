"""Policy controls surrounding raw solution-refinement indicators."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from aicfd.geometry import GeometryRefinementPolicy, Obstacle2D
from aicfd.geometry._cell import cell_polygon_2d
from aicfd.representation import AdaptiveTree, Cell


class LevelFloor(Protocol):
    """Callable that supplies a minimum permissible level at one cell."""

    def __call__(
        self,
        tree: AdaptiveTree,
        cell: Cell,
        *,
        for_coarsening: bool = False,
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class GeometryLevelFloor:
    """Adapt an obstacle policy to the generic solution-AMR level-floor API."""

    obstacle: Obstacle2D
    policy: GeometryRefinementPolicy

    def __call__(
        self,
        tree: AdaptiveTree,
        cell: Cell,
        *,
        for_coarsening: bool = False,
    ) -> int:
        return self.policy.target_level(
            tree,
            cell,
            self.obstacle,
            for_coarsening=for_coarsening,
        )

    def fluid_measure(self, tree: AdaptiveTree, cell: Cell) -> float:
        """Return clipped fluid area for conservative cut-cell transfer."""

        if tree.dimension != 2:
            raise ValueError("obstacle fluid measure currently requires a 2D tree")
        return float(cell_polygon_2d(tree, cell).difference(self.obstacle.polygon).area)


@dataclass(frozen=True, slots=True)
class SolutionRefinementPolicy:
    """Turn normalized scores into safe, deterministic topology changes."""

    min_level: int = 0
    max_level: int = 7
    refine_threshold: float = 1.0
    coarsen_threshold: float = 0.25
    buffer_layers: int = 1
    enforce_balance: bool = True
    max_cells: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("min_level", self.min_level),
            ("max_level", self.max_level),
            ("buffer_layers", self.buffer_layers),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.min_level > self.max_level:
            raise ValueError("min_level must not exceed max_level")

        if (
            not isfinite(self.refine_threshold)
            or not isfinite(self.coarsen_threshold)
            or self.refine_threshold <= 0.0
            or self.coarsen_threshold < 0.0
        ):
            raise ValueError("adaptation thresholds must be finite and non-negative")
        if self.coarsen_threshold >= self.refine_threshold:
            raise ValueError("coarsen_threshold must be below refine_threshold")
        if self.max_cells is not None:
            if not isinstance(self.max_cells, int) or isinstance(self.max_cells, bool):
                raise TypeError("max_cells must be an integer when provided")
            if self.max_cells < 1:
                raise ValueError("max_cells must be positive")
