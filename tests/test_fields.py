import numpy as np
import pytest

from aicfd import (
    AdaptiveTree,
    Cell,
    CellField,
    FieldLocation,
    FieldSpec,
    GeometryLevelFloor,
    GeometryRefinementPolicy,
    Obstacle2D,
    State,
    TreeLayout,
    ValueRepresentation,
    adapt_state_topology,
    field_total,
    measures_from_provider,
)


def _uniform_tree(dimension: int, level: int) -> AdaptiveTree:
    tree = AdaptiveTree(dimension)
    for _ in range(level):
        tree.refine_many(tree.leaves)
    return tree


def test_layout_is_a_deterministic_snapshot_not_cell_identity() -> None:
    tree = _uniform_tree(2, 1)
    layout = TreeLayout.from_tree(tree)

    assert layout.cells == tree.leaves
    assert [layout.index(cell) for cell in layout.cells] == list(range(4))
    assert layout.to_tree().leaves == tree.leaves

    tree.refine(tree.leaves[0])
    changed = TreeLayout.from_tree(tree)

    assert changed.topology_id != layout.topology_id
    assert len(layout) == 4


def test_cell_field_validates_shape_location_bounds_and_layout() -> None:
    tree = _uniform_tree(2, 1)
    layout = TreeLayout.from_tree(tree)
    spec = FieldSpec("fraction", lower_bound=0.0, upper_bound=1.0)
    field = CellField(spec, layout, [0.0, 0.25, 0.5, 1.0])

    assert field.at(layout.cells[2]) == 0.5
    assert field.layout_id == layout.topology_id

    with pytest.raises(ValueError, match="shape"):
        CellField(spec, layout, [0.0])
    with pytest.raises(ValueError, match="upper bound"):
        CellField(spec, layout, [0.0, 0.0, 0.0, 1.1])
    with pytest.raises(ValueError, match="cell-located"):
        CellField(
            FieldSpec("flux", location=FieldLocation.FACE_X),
            layout,
            np.zeros(4),
        )

    other_tree = AdaptiveTree(2)
    other_tree.refine(other_tree.root)
    other_tree.refine(other_tree.leaves[-1])
    other_layout = TreeLayout.from_tree(other_tree)
    with pytest.raises(ValueError, match="stale or different"):
        State(other_layout, [field])


def test_state_registry_rejects_duplicate_names_and_can_add_a_field() -> None:
    layout = TreeLayout.from_tree(AdaptiveTree(1))
    first = CellField(FieldSpec("phi"), layout, [2.0])
    state = State(layout, [first], time=0.5, step=3)
    second = CellField(FieldSpec("temperature", unit="K"), layout, [300.0])

    extended = state.with_field(second)

    assert tuple(extended.fields) == ("phi", "temperature")
    assert extended.time == 0.5
    assert extended.step == 3
    with pytest.raises(ValueError, match="already exists"):
        state.with_field(first)
    with pytest.raises(ValueError, match="duplicate"):
        State(layout, [first, first])


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_constant_cell_average_refines_and_coarsens_conservatively(
    dimension: int,
) -> None:
    tree = AdaptiveTree(dimension)
    layout = TreeLayout.from_tree(tree)
    state = State(layout, [CellField(FieldSpec("phi"), layout, [3.5])])

    refined, refine_report = adapt_state_topology(state, refine=[tree.root])

    assert refine_report.leaves_after == 2**dimension
    np.testing.assert_array_equal(refined["phi"].values, 3.5)
    assert field_total(refined["phi"]) == pytest.approx(field_total(state["phi"]))

    coarsened, coarsen_report = adapt_state_topology(
        refined,
        coarsen=[tree.root],
    )

    assert coarsen_report.leaves_after == 1
    np.testing.assert_array_equal(coarsened["phi"].values, [3.5])


def test_restriction_uses_volume_weighted_cell_averages() -> None:
    tree = _uniform_tree(2, 1)
    layout = TreeLayout.from_tree(tree)
    state = State(layout, [CellField(FieldSpec("phi"), layout, [1, 2, 3, 4])])

    coarsened, _ = adapt_state_topology(state, coarsen=[tree.root])

    assert coarsened["phi"].values == pytest.approx([2.5])
    assert field_total(coarsened["phi"]) == pytest.approx(field_total(state["phi"]))


def test_float32_transfer_uses_dtype_appropriate_conservation_tolerance() -> None:
    tree = _uniform_tree(2, 1)
    layout = TreeLayout.from_tree(tree)
    values = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    state = State(
        layout,
        [CellField(FieldSpec("phi", dtype="float32"), layout, values)],
    )

    coarsened, _ = adapt_state_topology(state, coarsen=[tree.root])

    assert coarsened["phi"].values.dtype == np.float32
    assert coarsened["phi"].values == pytest.approx([0.25])


def test_cell_integral_is_distributed_and_summed_exactly() -> None:
    tree = AdaptiveTree(2)
    layout = TreeLayout.from_tree(tree)
    spec = FieldSpec(
        "amount",
        value_representation=ValueRepresentation.CELL_INTEGRAL,
    )
    state = State(layout, [CellField(spec, layout, [8.0])])

    refined, _ = adapt_state_topology(state, refine=[tree.root])

    np.testing.assert_array_equal(refined["amount"].values, [2.0] * 4)
    coarsened, _ = adapt_state_topology(refined, coarsen=[tree.root])
    np.testing.assert_array_equal(coarsened["amount"].values, [8.0])


def test_cut_cell_fluid_measures_preserve_the_extensive_total() -> None:
    tree = AdaptiveTree(2)
    obstacle = Obstacle2D.circle((0.5, 0.5), 0.2, segments=96)
    geometry_floor = GeometryLevelFloor(
        obstacle,
        GeometryRefinementPolicy(
            min_level=0,
            boundary_level=0,
            max_level=1,
            feature_angle_degrees=None,
        ),
    )
    layout = TreeLayout.from_tree(tree)
    state = State(layout, [CellField(FieldSpec("phi"), layout, [2.0])])
    old_measures = measures_from_provider(tree, geometry_floor.fluid_measure)

    refined, _ = adapt_state_topology(
        state,
        refine=[tree.root],
        measure_provider=geometry_floor.fluid_measure,
    )
    new_tree = refined.to_tree()
    new_measures = measures_from_provider(new_tree, geometry_floor.fluid_measure)

    assert field_total(refined["phi"], new_measures) == pytest.approx(
        field_total(state["phi"], old_measures), rel=1.0e-12
    )
    np.testing.assert_array_equal(refined["phi"].values, [2.0] * 4)


def test_explicit_topology_requests_must_be_disjoint() -> None:
    tree = _uniform_tree(1, 1)
    layout = TreeLayout.from_tree(tree)
    state = State(layout, [CellField(FieldSpec("phi"), layout, [1.0, 1.0])])

    with pytest.raises(ValueError, match="disjoint"):
        adapt_state_topology(
            state,
            refine=[Cell(1, (0,))],
            coarsen=[tree.root],
        )
