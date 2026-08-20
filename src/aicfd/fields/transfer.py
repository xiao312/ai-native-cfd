"""Conservative field transfer between adaptive-tree snapshots."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from aicfd.fields.field import CellField, ValueRepresentation
from aicfd.fields.layout import TreeLayout
from aicfd.fields.state import State
from aicfd.representation import AdaptiveTree, Cell

CellMeasures = Mapping[Cell, float]
MeasureProvider = Callable[[AdaptiveTree, Cell], float]


def _measure_array(
    layout: TreeLayout,
    measures: CellMeasures | None,
) -> NDArray[np.float64]:
    if measures is None:
        return np.array(
            [layout.cell_measure(cell) for cell in layout.cells], dtype=np.float64
        )
    if set(measures) != set(layout.cells):
        raise ValueError("cell measures must contain exactly the layout leaves")
    values = np.array([measures[cell] for cell in layout.cells], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("cell measures must be finite and non-negative")
    return values


def measures_from_provider(
    tree: AdaptiveTree,
    provider: MeasureProvider,
) -> dict[Cell, float]:
    """Evaluate an effective-volume provider for every current leaf."""

    measures = {cell: float(provider(tree, cell)) for cell in tree.leaves}
    if any(not isfinite(value) or value < 0.0 for value in measures.values()):
        raise ValueError("measure provider returned an invalid cell measure")
    return measures


def field_total(
    field: CellField,
    measures: CellMeasures | None = None,
) -> float | NDArray[np.float64]:
    """Return the extensive total implied by a cell field.

    Cell averages are multiplied by effective cell measures, cell integrals are
    summed directly, and point samples have no meaningful extensive total.
    """

    representation = field.spec.value_representation
    if representation is ValueRepresentation.POINT_VALUE:
        raise ValueError("point values do not define an extensive total")

    rows = np.asarray(field.values, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[:, np.newaxis]
    if representation is ValueRepresentation.CELL_AVERAGE:
        weights = _measure_array(field.layout, measures)
        total = np.sum(rows * weights[:, np.newaxis], axis=0)
    else:
        total = np.sum(rows, axis=0)
    if field.spec.n_components == 1:
        return float(total[0])
    return total


def _related_old_cells(new_cell: Cell, old_cells: tuple[Cell, ...]) -> tuple[Cell, ...]:
    return tuple(cell for cell in old_cells if new_cell.is_ancestor_of(cell))


def _old_ancestor(new_cell: Cell, old_cells: tuple[Cell, ...]) -> Cell | None:
    for cell in old_cells:
        if cell.is_ancestor_of(new_cell):
            return cell
    return None


def _remap_values(
    field: CellField,
    new_layout: TreeLayout,
    old_measures: NDArray[np.float64],
    new_measures: NDArray[np.float64],
) -> NDArray[np.floating]:
    old_layout = field.layout
    old_rows = np.asarray(field.values)
    scalar = old_rows.ndim == 1
    if scalar:
        old_rows = old_rows[:, np.newaxis]
    result = np.empty(
        (len(new_layout), field.spec.n_components), dtype=field.spec.dtype
    )

    old_lookup = old_layout.cell_to_index
    new_lookup = new_layout.cell_to_index
    representation = field.spec.value_representation

    # For integral prolongation, each parent integral is distributed over all of
    # its new descendant leaves in proportion to their effective volumes.
    descendant_measure: dict[Cell, float] = {}
    descendant_count: dict[Cell, int] = {}
    if representation is ValueRepresentation.CELL_INTEGRAL:
        for old_cell in old_layout.cells:
            descendants = tuple(
                cell for cell in new_layout.cells if old_cell.is_ancestor_of(cell)
            )
            if descendants:
                descendant_measure[old_cell] = sum(
                    new_measures[new_lookup[cell]] for cell in descendants
                )
                descendant_count[old_cell] = len(descendants)

    for new_index, new_cell in enumerate(new_layout.cells):
        if new_cell in old_lookup:
            result[new_index] = old_rows[old_lookup[new_cell]]
            continue

        ancestor = _old_ancestor(new_cell, old_layout.cells)
        if ancestor is not None:
            source = old_rows[old_lookup[ancestor]]
            if representation is ValueRepresentation.CELL_INTEGRAL:
                denominator = descendant_measure[ancestor]
                if denominator > 0.0:
                    fraction = new_measures[new_index] / denominator
                else:
                    fraction = 1.0 / descendant_count[ancestor]
                result[new_index] = source * fraction
            else:
                # First-order prolongation for a cell average or point value.
                result[new_index] = source
            continue

        descendants = _related_old_cells(new_cell, old_layout.cells)
        if not descendants:
            raise RuntimeError(f"old and new layouts do not overlap at {new_cell}")
        descendant_indices = np.array(
            [old_lookup[cell] for cell in descendants], dtype=np.int64
        )
        values = old_rows[descendant_indices]
        if representation is ValueRepresentation.CELL_INTEGRAL:
            result[new_index] = np.sum(values, axis=0)
            continue

        weights = old_measures[descendant_indices]
        weight_sum = float(np.sum(weights))
        if weight_sum > 0.0:
            result[new_index] = (
                np.sum(values * weights[:, np.newaxis], axis=0) / weight_sum
            )
        else:
            # A completely solid parent contributes no conserved amount. An
            # unweighted mean keeps any placeholder value bounded and deterministic.
            result[new_index] = np.mean(values, axis=0)

    return result[:, 0] if scalar else result


def remap_state(
    state: State,
    new_tree: AdaptiveTree,
    *,
    old_measures: CellMeasures | None = None,
    new_measures: CellMeasures | None = None,
    check_conservation: bool = True,
) -> State:
    """Transfer every field to ``new_tree`` without mutating ``state``.

    Unchanged leaves are copied, descendants receive first-order prolongation,
    and new parents receive volume-weighted restriction. Optional measures allow
    callers to use cut-cell fluid volumes instead of full Cartesian volumes.
    """

    new_layout = TreeLayout.from_tree(new_tree)
    old_layout = state.layout
    if (
        new_layout.dimension != old_layout.dimension
        or new_layout.origin != old_layout.origin
        or new_layout.extent != old_layout.extent
    ):
        raise ValueError("field transfer requires the same physical domain")

    old_weights = _measure_array(old_layout, old_measures)
    new_weights = _measure_array(new_layout, new_measures)
    transferred: list[CellField] = []
    for field in state.fields.values():
        values = _remap_values(field, new_layout, old_weights, new_weights)
        new_field = CellField(field.spec, new_layout, values)
        if (
            check_conservation
            and field.spec.value_representation is not ValueRepresentation.POINT_VALUE
        ):
            before = np.asarray(field_total(field, old_measures), dtype=np.float64)
            after = np.asarray(field_total(new_field, new_measures), dtype=np.float64)
            scale = max(1.0, float(np.max(np.abs(before))))
            tolerance = max(1.0e-12, 32.0 * np.finfo(field.spec.dtype).eps)
            if not np.allclose(
                before,
                after,
                rtol=tolerance,
                atol=tolerance * scale,
            ):
                raise RuntimeError(
                    f"field transfer did not conserve {field.spec.name!r}: "
                    f"{before} != {after}"
                )
        transferred.append(new_field)

    return State(new_layout, transferred, time=state.time, step=state.step)


@dataclass(frozen=True, slots=True)
class TopologyAdaptationReport:
    """Operations performed by one explicit topology transaction."""

    leaves_before: int
    leaves_after: int
    refined_parents: tuple[Cell, ...]
    balance_refined_parents: tuple[Cell, ...]
    coarsened_parents: tuple[Cell, ...]


def adapt_state_topology(
    state: State,
    *,
    refine: Iterable[Cell] = (),
    coarsen: Iterable[Cell] = (),
    enforce_balance: bool = True,
    measure_provider: MeasureProvider | None = None,
) -> tuple[State, TopologyAdaptationReport]:
    """Apply explicit, disjoint tree requests and transfer all fields atomically."""

    old_tree = state.to_tree()
    new_tree = state.to_tree()
    refine_cells = tuple(dict.fromkeys(refine))
    coarsen_parents = tuple(dict.fromkeys(coarsen))

    for parent in coarsen_parents:
        if any(parent == cell or parent.is_ancestor_of(cell) for cell in refine_cells):
            raise ValueError("refinement and coarsening requests must be disjoint")
    for first_index, first in enumerate(coarsen_parents):
        if any(
            first.is_ancestor_of(second) or second.is_ancestor_of(first)
            for second in coarsen_parents[first_index + 1 :]
        ):
            raise ValueError("coarsening requests must not overlap")

    completed_coarsening: list[Cell] = []
    for parent in sorted(coarsen_parents, key=lambda cell: -cell.level):
        new_tree.coarsen(parent)
        completed_coarsening.append(parent)

    completed_refinement: list[Cell] = []
    for cell in refine_cells:
        new_tree.refine(cell)
        completed_refinement.append(cell)

    balance_refined = new_tree.balance() if enforce_balance else ()
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
    return new_state, TopologyAdaptationReport(
        leaves_before=len(state.layout),
        leaves_after=len(new_state.layout),
        refined_parents=tuple(completed_refinement),
        balance_refined_parents=tuple(balance_refined),
        coarsened_parents=tuple(completed_coarsening),
    )
