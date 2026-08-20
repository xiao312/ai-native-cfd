"""Transactional solution-driven refinement and conservative field transfer."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from aicfd.amr.indicators import Indicator, IndicatorResult, combine_indicators
from aicfd.amr.policy import LevelFloor, SolutionRefinementPolicy
from aicfd.fields import (
    MeasureProvider,
    State,
    measures_from_provider,
    remap_state,
)
from aicfd.representation import AdaptiveTree, Cell


@dataclass(frozen=True, slots=True)
class SolutionAdaptationReport:
    """Audit trail for one solution-driven adaptation transaction."""

    leaves_before: int
    leaves_after: int
    indicator_names: tuple[str, ...]
    maximum_score: float
    coarsened_parents: tuple[Cell, ...]
    floor_refined_parents: tuple[Cell, ...]
    indicator_refined_parents: tuple[Cell, ...]
    balance_refined_parents: tuple[Cell, ...]
    budget_skipped_parents: tuple[Cell, ...]

    @property
    def refinement_count(self) -> int:
        return (
            len(self.floor_refined_parents)
            + len(self.indicator_refined_parents)
            + len(self.balance_refined_parents)
        )

    @property
    def coarsening_count(self) -> int:
        return len(self.coarsened_parents)


def _combined_scores(
    state: State,
    results: tuple[IndicatorResult, ...],
) -> np.ndarray:
    if not results:
        return np.zeros(len(state.layout), dtype=np.float64)
    return np.max(np.stack([result.scores for result in results]), axis=0)


def _buffered_priorities(
    tree: AdaptiveTree,
    seeds: dict[Cell, float],
    layers: int,
) -> dict[Cell, float]:
    priorities = dict(seeds)
    frontier = dict(seeds)
    for _ in range(layers):
        next_frontier: dict[Cell, float] = {}
        for cell, priority in frontier.items():
            for neighbor in tree.face_neighbors(cell):
                if priority > priorities.get(neighbor, -1.0):
                    priorities[neighbor] = priority
                    next_frontier[neighbor] = priority
        frontier = next_frontier
        if not frontier:
            break
    return priorities


def _required_level(
    tree: AdaptiveTree,
    cell: Cell,
    policy: SolutionRefinementPolicy,
    level_floor: LevelFloor | None,
    *,
    for_coarsening: bool = False,
) -> int:
    required = policy.min_level
    if level_floor is not None:
        required = max(
            required,
            level_floor(tree, cell, for_coarsening=for_coarsening),
        )
    if required > policy.max_level:
        raise ValueError(
            f"required level {required} exceeds solution policy max_level "
            f"{policy.max_level}"
        )
    return required


def _enforce_level_floor(
    tree: AdaptiveTree,
    policy: SolutionRefinementPolicy,
    level_floor: LevelFloor | None,
) -> list[Cell]:
    refined: list[Cell] = []
    while True:
        targets = tuple(
            cell
            for cell in tree.leaves
            if cell.level < _required_level(tree, cell, policy, level_floor)
        )
        if not targets:
            return refined
        for cell in targets:
            tree.refine(cell)
            refined.append(cell)


def _coarsen_safe_families(
    tree: AdaptiveTree,
    scores: dict[Cell, float],
    protected: set[Cell],
    policy: SolutionRefinementPolicy,
    level_floor: LevelFloor | None,
) -> list[Cell]:
    candidates = {
        parent
        for leaf in tree.leaves
        if (parent := leaf.parent) is not None and tree.can_coarsen(parent)
    }
    accepted: list[Cell] = []
    for parent in sorted(candidates, key=lambda cell: (-cell.level, cell.index)):
        children = parent.children()
        if any(child in protected for child in children):
            continue
        if any(scores[child] > policy.coarsen_threshold for child in children):
            continue
        if parent.level < _required_level(
            tree,
            parent,
            policy,
            level_floor,
            for_coarsening=True,
        ):
            continue
        tree.coarsen(parent)
        accepted.append(parent)
    return accepted


def _candidate_key(
    cell: Cell,
    priorities: dict[Cell, float],
    seeds: set[Cell],
) -> tuple[float | int | tuple[int, ...], ...]:
    return (
        0 if cell in seeds else 1,
        -priorities[cell],
        cell.level,
        cell.index,
    )


def _refine_indicator_candidates(
    tree: AdaptiveTree,
    candidates: Iterable[Cell],
    priorities: dict[Cell, float],
    seeds: set[Cell],
    policy: SolutionRefinementPolicy,
) -> tuple[list[Cell], list[Cell], list[Cell], AdaptiveTree]:
    refined: list[Cell] = []
    balance_refined: list[Cell] = []
    budget_skipped: list[Cell] = []
    ordered = sorted(
        candidates,
        key=lambda cell: _candidate_key(cell, priorities, seeds),
    )

    for cell in ordered:
        if cell not in tree or cell.level >= policy.max_level:
            continue
        if policy.max_cells is None:
            tree.refine(cell)
            refined.append(cell)
            continue

        trial = AdaptiveTree.from_dict(tree.to_dict())
        trial.refine(cell)
        trial_balance = trial.balance() if policy.enforce_balance else ()
        if len(trial) > policy.max_cells:
            budget_skipped.append(cell)
            continue
        tree = trial
        refined.append(cell)
        balance_refined.extend(trial_balance)

    # Assignment to a local trial tree cannot replace the caller's object, so the
    # accepted tree is part of the return value.
    return refined, balance_refined, budget_skipped, tree


def adapt_solution(
    state: State,
    indicators: Iterable[Indicator],
    policy: SolutionRefinementPolicy,
    *,
    level_floor: LevelFloor | None = None,
    measure_provider: MeasureProvider | None = None,
    allow_coarsening: bool = True,
) -> tuple[State, SolutionAdaptationReport]:
    """Adapt a copied tree, conservatively transfer fields, then publish state.

    Indicators are evaluated only on the old immutable layout. The input state is
    unchanged if topology generation, transfer, or conservation validation fails.
    """

    indicator_tuple = tuple(indicators)
    results = combine_indicators(state, indicator_tuple)
    combined = _combined_scores(state, results)
    scores = {
        cell: float(combined[state.layout.index(cell)]) for cell in state.layout.cells
    }
    seeds = {
        cell: score
        for cell, score in scores.items()
        if score >= policy.refine_threshold and cell.level < policy.max_level
    }

    old_tree = state.to_tree()
    if old_tree.max_level > policy.max_level:
        raise ValueError("current tree is deeper than solution policy max_level")
    priorities = _buffered_priorities(old_tree, seeds, policy.buffer_layers)
    protected = set(priorities)

    new_tree = state.to_tree()
    coarsened = (
        _coarsen_safe_families(
            new_tree,
            scores,
            protected,
            policy,
            level_floor,
        )
        if allow_coarsening
        else []
    )
    floor_refined = _enforce_level_floor(new_tree, policy, level_floor)
    floor_balance = list(new_tree.balance()) if policy.enforce_balance else []
    if policy.max_cells is not None and len(new_tree) > policy.max_cells:
        raise ValueError("required levels and 2:1 balance exceed max_cells")

    candidates = tuple(
        cell
        for cell in priorities
        if cell in new_tree and cell.level < policy.max_level
    )
    (
        indicator_refined,
        candidate_balance,
        budget_skipped,
        new_tree,
    ) = _refine_indicator_candidates(
        new_tree,
        candidates,
        priorities,
        set(seeds),
        policy,
    )
    final_balance = list(new_tree.balance()) if policy.enforce_balance else []
    all_balance = floor_balance + candidate_balance + final_balance
    if policy.max_cells is not None and len(new_tree) > policy.max_cells:
        raise RuntimeError("adaptation exceeded max_cells after balancing")
    new_tree.assert_valid()

    old_measures = (
        None
        if measure_provider is None
        else measures_from_provider(old_tree, measure_provider)
    )
    new_measures = (
        None
        if measure_provider is None
        else measures_from_provider(new_tree, measure_provider)
    )
    new_state = remap_state(
        state,
        new_tree,
        old_measures=old_measures,
        new_measures=new_measures,
    )
    return new_state, SolutionAdaptationReport(
        leaves_before=len(state.layout),
        leaves_after=len(new_state.layout),
        indicator_names=tuple(result.name for result in results),
        maximum_score=float(np.max(combined)) if len(combined) else 0.0,
        coarsened_parents=tuple(coarsened),
        floor_refined_parents=tuple(floor_refined),
        indicator_refined_parents=tuple(indicator_refined),
        balance_refined_parents=tuple(all_balance),
        budget_skipped_parents=tuple(budget_skipped),
    )
