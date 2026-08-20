"""Geometry-driven refinement and coarsening of a quadtree."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, degrees, hypot, log2, sqrt

from aicfd.geometry._cell import cell_polygon_2d
from aicfd.geometry.obstacle import Obstacle2D
from aicfd.representation import AdaptiveTree, Cell


@dataclass(frozen=True, slots=True)
class DistanceBand:
    """Require at least ``level`` within ``distance`` of the obstacle."""

    distance: float
    level: int

    def __post_init__(self) -> None:
        if self.distance < 0.0:
            raise ValueError("a distance-band radius must be non-negative")
        if not isinstance(self.level, int) or isinstance(self.level, bool):
            raise TypeError("a distance-band level must be an integer")
        if self.level < 0:
            raise ValueError("a distance-band level must be non-negative")


@dataclass(frozen=True, slots=True)
class GeometryRefinementPolicy:
    """Convert local obstacle characteristics into a desired quadtree level.

    The desired level is the maximum requested by five independent rules:

    * a minimum level for the overall domain;
    * a minimum level for cells cut by the obstacle boundary;
    * user-visible distance bands around the boundary;
    * a chord-error estimate derived from local boundary curvature;
    * an optional extra level around sharp polyline vertices.

    A cell can be coarsened only when its parent still satisfies these rules.
    ``coarsening_hysteresis`` expands distance/feature bands during that check,
    avoiding immediate refine/coarsen oscillation when a boundary moves slightly.
    """

    min_level: int = 1
    boundary_level: int = 4
    max_level: int = 7
    distance_bands: tuple[DistanceBand, ...] = ()
    max_chord_error: float | None = None
    feature_angle_degrees: float | None = 45.0
    feature_level: int | None = None
    feature_distance: float = 0.0
    coarsening_hysteresis: float = 0.15
    enforce_balance: bool = True

    def __post_init__(self) -> None:
        integer_fields = {
            "min_level": self.min_level,
            "boundary_level": self.boundary_level,
            "max_level": self.max_level,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if not 0 <= self.min_level <= self.boundary_level <= self.max_level:
            raise ValueError(
                "levels must satisfy min_level <= boundary_level <= max_level"
            )

        bands = tuple(sorted(self.distance_bands, key=lambda band: band.distance))
        if any(not self.min_level <= band.level <= self.max_level for band in bands):
            raise ValueError("distance-band levels must lie within policy levels")
        object.__setattr__(self, "distance_bands", bands)

        if self.max_chord_error is not None and self.max_chord_error <= 0.0:
            raise ValueError("max_chord_error must be positive when provided")
        if self.feature_angle_degrees is not None and not (
            0.0 <= self.feature_angle_degrees <= 180.0
        ):
            raise ValueError("feature_angle_degrees must lie between 0 and 180")
        if self.feature_level is not None and not (
            self.boundary_level <= self.feature_level <= self.max_level
        ):
            raise ValueError(
                "feature_level must lie between boundary_level and max_level"
            )
        if self.feature_distance < 0.0:
            raise ValueError("feature_distance must be non-negative")
        if self.coarsening_hysteresis < 0.0:
            raise ValueError("coarsening_hysteresis must be non-negative")

    def _level_for_size(self, tree: AdaptiveTree, target_size: float) -> int:
        ratio = max(tree.extent) / target_size
        level = 0 if ratio <= 1.0 else ceil(log2(ratio))
        return max(self.min_level, min(self.max_level, level))

    def target_level(
        self,
        tree: AdaptiveTree,
        cell: Cell,
        obstacle: Obstacle2D,
        *,
        for_coarsening: bool = False,
    ) -> int:
        """Return the minimum acceptable level for ``cell`` at its location."""

        cell_shape = cell_polygon_2d(tree, cell)

        # Cells completely inside the solid do not participate in a fluid solve.
        # Keeping their interiors coarse is one benefit of an embedded boundary.
        if obstacle.polygon.covers(cell_shape):
            return self.min_level

        target = self.min_level
        expansion = 1.0 + self.coarsening_hysteresis if for_coarsening else 1.0
        boundary_distance = cell_shape.distance(obstacle.boundary)
        for band in self.distance_bands:
            if boundary_distance <= expansion * band.distance:
                target = max(target, band.level)

        intersects_boundary = cell_shape.intersects(obstacle.boundary)
        if intersects_boundary:
            target = max(target, self.boundary_level)

            if self.max_chord_error is not None:
                width_x, width_y = tree.cell_size(cell)
                search_distance = 0.5 * hypot(width_x, width_y)
                local_features = obstacle.features_near(cell_shape, search_distance)
                if local_features:
                    maximum_curvature = max(
                        feature.curvature for feature in local_features
                    )
                    if maximum_curvature > 0.0:
                        # For a short circular arc, sagitta ~= curvature*h^2/8.
                        target_size = sqrt(
                            8.0 * self.max_chord_error / maximum_curvature
                        )
                        target = max(target, self._level_for_size(tree, target_size))

        if self.feature_level is not None and self.feature_angle_degrees is not None:
            feature_distance = expansion * self.feature_distance
            nearby_features = obstacle.features_near(cell_shape, feature_distance)
            if any(
                degrees(feature.turning_angle) >= self.feature_angle_degrees
                for feature in nearby_features
            ):
                target = max(target, self.feature_level)

        return min(target, self.max_level)


@dataclass(frozen=True, slots=True)
class AdaptationReport:
    """A compact audit trail for one geometry adaptation pass."""

    leaves_before: int
    leaves_after: int
    refined_parents: tuple[Cell, ...]
    balance_refined_parents: tuple[Cell, ...]
    coarsened_parents: tuple[Cell, ...]

    @property
    def refinement_count(self) -> int:
        """Number of explicit and balancing refinement operations."""

        return len(self.refined_parents) + len(self.balance_refined_parents)

    @property
    def coarsening_count(self) -> int:
        """Number of sibling families replaced by their parent."""

        return len(self.coarsened_parents)


def adapt_to_obstacle(
    tree: AdaptiveTree,
    obstacle: Obstacle2D,
    policy: GeometryRefinementPolicy,
    *,
    allow_coarsening: bool = True,
) -> AdaptationReport:
    """Refine and safely coarsen a 2D tree to meet a geometry policy."""

    if tree.dimension != 2:
        raise ValueError("obstacle adaptation currently requires a 2D tree")

    leaves_before = len(tree)
    refined: list[Cell] = []
    while True:
        targets = tuple(
            cell
            for cell in tree.leaves
            if cell.level < policy.target_level(tree, cell, obstacle)
        )
        if not targets:
            break
        for cell in targets:
            tree.refine(cell)
            refined.append(cell)

    balance_refined = list(tree.balance()) if policy.enforce_balance else []
    coarsened: list[Cell] = []
    if allow_coarsening:
        while True:
            candidates = {
                parent
                for leaf in tree.leaves
                if (parent := leaf.parent) is not None and tree.can_coarsen(parent)
            }
            accepted_in_pass = False
            ordered_candidates = sorted(
                candidates, key=lambda cell: (-cell.level, cell.index)
            )
            for parent in ordered_candidates:
                if not tree.can_coarsen(parent):
                    continue
                target = policy.target_level(
                    tree, parent, obstacle, for_coarsening=True
                )
                if target > parent.level:
                    continue

                tree.coarsen(parent)
                if policy.enforce_balance and not tree.is_balanced():
                    tree.refine(parent)
                    continue
                coarsened.append(parent)
                accepted_in_pass = True

            if not accepted_in_pass:
                break

    tree.assert_valid()
    return AdaptationReport(
        leaves_before=leaves_before,
        leaves_after=len(tree),
        refined_parents=tuple(refined),
        balance_refined_parents=tuple(balance_refined),
        coarsened_parents=tuple(coarsened),
    )
