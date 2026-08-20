"""Deterministic solution-driven adaptive mesh refinement."""

from aicfd.amr.adaptation import SolutionAdaptationReport, adapt_solution
from aicfd.amr.indicators import (
    Indicator,
    IndicatorResult,
    NeighborJumpIndicator,
    ValueRangeIndicator,
    WaveletDetailIndicator,
    combine_indicators,
)
from aicfd.amr.policy import (
    GeometryLevelFloor,
    LevelFloor,
    SolutionRefinementPolicy,
)

__all__ = [
    "GeometryLevelFloor",
    "Indicator",
    "IndicatorResult",
    "LevelFloor",
    "NeighborJumpIndicator",
    "SolutionAdaptationReport",
    "SolutionRefinementPolicy",
    "ValueRangeIndicator",
    "WaveletDetailIndicator",
    "adapt_solution",
    "combine_indicators",
]
