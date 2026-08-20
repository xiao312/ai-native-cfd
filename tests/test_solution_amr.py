import math

import numpy as np
import pytest

from aicfd import (
    AdaptiveTree,
    Cell,
    CellField,
    FieldSpec,
    GeometryLevelFloor,
    GeometryRefinementPolicy,
    NeighborJumpIndicator,
    Obstacle2D,
    SolutionRefinementPolicy,
    State,
    TreeLayout,
    ValueRangeIndicator,
    WaveletDetailIndicator,
    adapt_solution,
    field_total,
    measures_from_provider,
    snap_to_obstacle,
)


def _uniform_tree(dimension: int, level: int) -> AdaptiveTree:
    tree = AdaptiveTree(dimension)
    for _ in range(level):
        tree.refine_many(tree.leaves)
    return tree


def _scalar_state(tree: AdaptiveTree, values) -> State:
    layout = TreeLayout.from_tree(tree)
    return State(layout, [CellField(FieldSpec("phi"), layout, values)])


def test_value_range_indicator_has_obvious_binary_scores() -> None:
    state = _scalar_state(_uniform_tree(1, 2), [0.0, 0.4, 0.6, 1.0])

    result = ValueRangeIndicator("phi", lower=0.25, upper=0.75).evaluate(state)

    np.testing.assert_array_equal(result.scores, [0.0, 1.0, 1.0, 0.0])


def test_neighbor_jump_is_normalized_by_a_physical_field_scale() -> None:
    state = _scalar_state(_uniform_tree(1, 2), [0.0, 0.0, 1.0, 1.0])

    result = NeighborJumpIndicator("phi", scale=0.5).evaluate(state)

    np.testing.assert_array_equal(result.scores, [0.0, 2.0, 2.0, 0.0])


def test_wavelet_detail_is_zero_for_constant_and_nonzero_for_varying_data() -> None:
    tree = _uniform_tree(2, 2)
    constant = _scalar_state(tree, np.ones(len(tree)))
    varying = _scalar_state(
        tree,
        [tree.cell_center(cell)[0] for cell in tree.leaves],
    )
    indicator = WaveletDetailIndicator("phi", scale=0.1)

    np.testing.assert_array_equal(indicator.evaluate(constant).scores, 0.0)
    assert np.max(indicator.evaluate(varying).scores) > 0.0


def test_solution_adaptation_refines_a_manufactured_gaussian_conservatively() -> None:
    tree = _uniform_tree(2, 2)
    values = []
    for cell in tree.leaves:
        x, y = tree.cell_center(cell)
        values.append(math.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.04))
    state = _scalar_state(tree, values)
    total_before = field_total(state["phi"])

    adapted, report = adapt_solution(
        state,
        [ValueRangeIndicator("phi", lower=0.2)],
        SolutionRefinementPolicy(
            min_level=2,
            max_level=3,
            refine_threshold=1.0,
            coarsen_threshold=0.25,
            buffer_layers=0,
        ),
        allow_coarsening=False,
    )

    assert report.indicator_refined_parents
    assert adapted.to_tree().max_level == 3
    assert adapted.to_tree().is_balanced()
    assert field_total(adapted["phi"]) == pytest.approx(total_before)
    assert state.to_tree().max_level == 2


def test_low_scores_coarsen_only_one_existing_sibling_generation() -> None:
    tree = _uniform_tree(2, 2)
    state = _scalar_state(tree, np.zeros(len(tree)))

    adapted, report = adapt_solution(
        state,
        [ValueRangeIndicator("phi", lower=1.0)],
        SolutionRefinementPolicy(
            min_level=1,
            max_level=3,
            refine_threshold=1.0,
            coarsen_threshold=0.25,
            buffer_layers=0,
        ),
    )

    assert report.coarsening_count == 4
    assert len(adapted.layout) == 4
    assert {cell.level for cell in adapted.layout.cells} == {1}


def test_buffer_layers_refine_face_neighbors_of_a_tagged_cell() -> None:
    tree = _uniform_tree(2, 2)
    selected = Cell(2, (1, 1))
    values = np.zeros(len(tree))
    values[TreeLayout.from_tree(tree).index(selected)] = 1.0
    state = _scalar_state(tree, values)

    adapted, report = adapt_solution(
        state,
        [ValueRangeIndicator("phi", lower=0.5)],
        SolutionRefinementPolicy(
            min_level=2,
            max_level=3,
            refine_threshold=1.0,
            coarsen_threshold=0.25,
            buffer_layers=1,
        ),
        allow_coarsening=False,
    )

    assert len(report.indicator_refined_parents) == 5
    assert selected in report.indicator_refined_parents
    assert len(adapted.layout) == 16 + 5 * 3


def test_cell_budget_keeps_high_priority_seed_before_other_cells() -> None:
    tree = _uniform_tree(2, 1)
    state = _scalar_state(tree, np.ones(len(tree)))

    adapted, report = adapt_solution(
        state,
        [ValueRangeIndicator("phi", lower=0.5)],
        SolutionRefinementPolicy(
            min_level=1,
            max_level=2,
            refine_threshold=1.0,
            coarsen_threshold=0.25,
            buffer_layers=0,
            max_cells=7,
        ),
        allow_coarsening=False,
    )

    assert len(adapted.layout) == 7
    assert report.indicator_refined_parents == (Cell(1, (0, 0)),)
    assert len(report.budget_skipped_parents) == 3


def test_cell_budget_accounts_for_balance_refinements() -> None:
    tree = AdaptiveTree(2)
    tree.refine(tree.root)
    tree.refine(Cell(1, (0, 0)))
    selected = Cell(2, (1, 0))
    layout = TreeLayout.from_tree(tree)
    values = np.zeros(len(tree))
    values[layout.index(selected)] = 1.0
    state = _scalar_state(tree, values)

    adapted, report = adapt_solution(
        state,
        [ValueRangeIndicator("phi", lower=0.5)],
        SolutionRefinementPolicy(
            min_level=1,
            max_level=3,
            refine_threshold=1.0,
            coarsen_threshold=0.25,
            buffer_layers=0,
            max_cells=10,
        ),
        allow_coarsening=False,
    )

    # Refining the level-2 seed adds three leaves, but maintaining 2:1 balance
    # would refine its level-1 neighbour and add three more. The whole proposal
    # is rejected rather than exceeding the ten-cell budget.
    assert len(adapted.layout) == 7
    assert report.indicator_refined_parents == ()
    assert report.budget_skipped_parents == (selected,)


def test_geometry_floor_and_cut_measures_combine_with_solution_amr() -> None:
    tree = AdaptiveTree(2)
    obstacle = Obstacle2D.circle((0.5, 0.5), 0.2, segments=96)
    geometry_floor = GeometryLevelFloor(
        obstacle,
        GeometryRefinementPolicy(
            min_level=0,
            boundary_level=2,
            max_level=2,
            feature_angle_degrees=None,
        ),
    )
    state = _scalar_state(tree, [2.0])
    old_measures = measures_from_provider(tree, geometry_floor.fluid_measure)

    adapted, report = adapt_solution(
        state,
        [],
        SolutionRefinementPolicy(
            min_level=0,
            max_level=2,
            refine_threshold=1.0,
            coarsen_threshold=0.25,
            buffer_layers=0,
        ),
        level_floor=geometry_floor,
        measure_provider=geometry_floor.fluid_measure,
    )
    new_tree = adapted.to_tree()
    snapped = snap_to_obstacle(new_tree, obstacle)
    new_measures = measures_from_provider(new_tree, geometry_floor.fluid_measure)

    assert report.floor_refined_parents
    assert all(cell.cell.level == 2 for cell in snapped.cut_cells)
    assert field_total(adapted["phi"], new_measures) == pytest.approx(
        field_total(state["phi"], old_measures), rel=1.0e-12
    )
