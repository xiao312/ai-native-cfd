import json

import numpy as np
import pytest

from aicfd import (
    AdaptiveTree,
    CellField,
    FieldSpec,
    State,
    TreeLayout,
    load_checkpoint,
    write_checkpoint,
)


def test_checkpoint_round_trip_restores_topology_metadata_and_arrays(tmp_path) -> None:
    tree = AdaptiveTree(2, origin=(-1.0, -2.0), extent=(3.0, 4.0))
    tree.refine(tree.root)
    layout = TreeLayout.from_tree(tree)
    pressure = CellField(
        FieldSpec("pressure", dtype="float32", unit="Pa"),
        layout,
        [1.0, 2.0, 3.0, 4.0],
    )
    velocity = CellField(
        FieldSpec(
            "velocity",
            component_names=("x", "y"),
            unit="m/s",
            unit_dimensions=(1, 0, -1, 0, 0, 0, 0),
        ),
        layout,
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
    )
    state = State(layout, [pressure, velocity], time=0.125, step=7)

    checkpoint = write_checkpoint(tmp_path / "chk00007", state)
    restored = load_checkpoint(checkpoint)

    assert restored.layout.topology_id == state.layout.topology_id
    assert restored.layout.cells == state.layout.cells
    assert restored.time == 0.125
    assert restored.step == 7
    assert restored["pressure"].spec == pressure.spec
    assert restored["velocity"].spec == velocity.spec
    np.testing.assert_array_equal(restored["pressure"].values, pressure.values)
    np.testing.assert_array_equal(restored["velocity"].values, velocity.values)


def test_checkpoint_never_overwrites_an_existing_path(tmp_path) -> None:
    layout = TreeLayout.from_tree(AdaptiveTree(1))
    state = State(layout, [CellField(FieldSpec("phi"), layout, [1.0])])
    target = tmp_path / "checkpoint"
    write_checkpoint(target, state)

    with pytest.raises(FileExistsError, match="already exists"):
        write_checkpoint(target, state)


def test_checkpoint_detects_a_tampered_topology_fingerprint(tmp_path) -> None:
    layout = TreeLayout.from_tree(AdaptiveTree(1))
    state = State(layout, [CellField(FieldSpec("phi"), layout, [1.0])])
    target = write_checkpoint(tmp_path / "checkpoint", state)
    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["topology_id"] = "not-the-real-fingerprint"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        load_checkpoint(target)
