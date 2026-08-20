"""Deterministic indicators that score leaves for solution-driven AMR."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from aicfd.fields import State, TreeLayout, ValueRepresentation
from aicfd.representation import Cell


@dataclass(frozen=True, slots=True, init=False)
class IndicatorResult:
    """One finite, non-negative score for every leaf in a layout."""

    name: str
    layout: TreeLayout
    scores: NDArray[np.float64]

    def __init__(self, name: str, layout: TreeLayout, scores: NDArray[np.float64]):
        values = np.array(scores, dtype=np.float64, copy=True)
        if not name:
            raise ValueError("indicator result needs a non-empty name")
        if values.shape != (len(layout),):
            raise ValueError("indicator result needs exactly one score per leaf")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("indicator scores must be finite and non-negative")
        values.setflags(write=False)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "scores", values)

    @property
    def layout_id(self) -> str:
        return self.layout.topology_id

    def at(self, cell: Cell) -> float:
        """Return the score attached to one leaf cell."""

        return float(self.scores[self.layout.index(cell)])


class Indicator(Protocol):
    """Protocol implemented by all solution-driven refinement indicators."""

    @property
    def name(self) -> str: ...

    def evaluate(self, state: State) -> IndicatorResult: ...


def _require_scale(scale: float) -> float:
    scale = float(scale)
    if not isfinite(scale) or scale <= 0.0:
        raise ValueError("indicator scale must be finite and positive")
    return scale


@dataclass(frozen=True, slots=True)
class ValueRangeIndicator:
    """Return one where a selected field component lies in a value interval."""

    field_name: str
    lower: float | None = None
    upper: float | None = None
    component: int | str = 0

    def __post_init__(self) -> None:
        if self.lower is None and self.upper is None:
            raise ValueError("a value-range indicator needs at least one bound")
        for bound in (self.lower, self.upper):
            if bound is not None and not isfinite(bound):
                raise ValueError("value-range bounds must be finite")
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower > self.upper
        ):
            raise ValueError("lower range bound must not exceed upper bound")

    @property
    def name(self) -> str:
        return f"value_range:{self.field_name}"

    def evaluate(self, state: State) -> IndicatorResult:
        values = state[self.field_name].component(self.component)
        selected = np.ones(len(state.layout), dtype=bool)
        if self.lower is not None:
            selected &= values >= self.lower
        if self.upper is not None:
            selected &= values <= self.upper
        return IndicatorResult(self.name, state.layout, selected.astype(np.float64))


@dataclass(frozen=True, slots=True)
class NeighborJumpIndicator:
    """Measure the largest normalized value jump across each leaf face."""

    field_name: str
    scale: float
    component: int | str = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale", _require_scale(self.scale))

    @property
    def name(self) -> str:
        return f"neighbor_jump:{self.field_name}"

    def evaluate(self, state: State) -> IndicatorResult:
        tree = state.to_tree()
        field = state[self.field_name]
        values = field.component(self.component)
        scores = np.zeros(len(state.layout), dtype=np.float64)
        for cell in state.layout.cells:
            row = state.layout.index(cell)
            scores[row] = max(
                (
                    abs(values[state.layout.index(neighbor)] - values[row]) / self.scale
                    for neighbor in tree.face_neighbors(cell)
                ),
                default=0.0,
            )
        return IndicatorResult(self.name, state.layout, scores)


@dataclass(frozen=True, slots=True)
class WaveletDetailIndicator:
    """A first-order Haar-like coarsen-and-predict defect on the tree.

    For each leaf, values under its parent are volume-averaged to the parent and
    prolonged back as a constant. The difference is the detail lost by that
    coarser representation. Level-zero cells have no parent and receive zero.
    """

    field_name: str
    scale: float
    component: int | str = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale", _require_scale(self.scale))

    @property
    def name(self) -> str:
        return f"wavelet_detail:{self.field_name}"

    def evaluate(self, state: State) -> IndicatorResult:
        field = state[self.field_name]
        if field.spec.value_representation is not ValueRepresentation.CELL_AVERAGE:
            raise ValueError("wavelet detail currently requires cell-average data")

        values = field.component(self.component)
        measures = np.array(
            [state.layout.cell_measure(cell) for cell in state.layout.cells],
            dtype=np.float64,
        )
        scores = np.zeros(len(state.layout), dtype=np.float64)
        parent_average: dict[Cell, float] = {}
        for cell in state.layout.cells:
            parent = cell.parent
            if parent is None:
                continue
            if parent not in parent_average:
                descendants = tuple(
                    leaf
                    for leaf in state.layout.cells
                    if parent == leaf or parent.is_ancestor_of(leaf)
                )
                rows = np.array(
                    [state.layout.index(leaf) for leaf in descendants],
                    dtype=np.int64,
                )
                weights = measures[rows]
                parent_average[parent] = float(
                    np.sum(values[rows] * weights) / np.sum(weights)
                )
            row = state.layout.index(cell)
            scores[row] = abs(values[row] - parent_average[parent]) / self.scale
        return IndicatorResult(self.name, state.layout, scores)


def combine_indicators(
    state: State,
    indicators: tuple[Indicator, ...],
) -> tuple[IndicatorResult, ...]:
    """Evaluate indicators and reject stale or mismatched results."""

    results = tuple(indicator.evaluate(state) for indicator in indicators)
    if any(result.layout_id != state.layout.topology_id for result in results):
        raise ValueError("an indicator returned scores for a stale tree layout")
    return results
