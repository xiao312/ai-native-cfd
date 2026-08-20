import json

import numpy as np
import pytest

from aicfd import (
    AdaptiveTree,
    FieldSpec,
    Trajectory,
    TrajectoryField,
    TreeLayout,
    load_trajectory,
    write_trajectory,
)


def _trajectory() -> Trajectory:
    tree = AdaptiveTree(2, origin=(-1.0, 2.0), extent=(3.0, 4.0))
    tree.refine(tree.root)
    layout = TreeLayout.from_tree(tree)
    scalar = TrajectoryField(
        FieldSpec("temperature", unit="K"),
        layout,
        [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
    )
    velocity = TrajectoryField(
        FieldSpec("U", component_names=("x", "y"), unit="m/s"),
        layout,
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
            [[2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
        ],
    )
    return Trajectory(
        layout,
        [0.0, 0.1],
        [scalar, velocity],
        steps=[0, 2],
        attributes={"source": "unit test"},
    )


def test_trajectory_materializes_states_without_sharing_field_arrays() -> None:
    trajectory = _trajectory()

    state = trajectory.state(1)

    assert state.time == pytest.approx(0.1)
    assert state.step == 2
    np.testing.assert_array_equal(
        state["temperature"].values,
        trajectory["temperature"].values[1],
    )
    state["temperature"].values[0] = -100.0
    assert trajectory["temperature"].values[1, 0] == 2.0


def test_trajectory_round_trip_preserves_layout_fields_and_attributes(tmp_path) -> None:
    trajectory = _trajectory()

    output = write_trajectory(tmp_path / "trajectory", trajectory)
    restored = load_trajectory(output)

    assert restored.layout.topology_id == trajectory.layout.topology_id
    assert dict(restored.attributes) == {"source": "unit test"}
    np.testing.assert_array_equal(restored.times, trajectory.times)
    np.testing.assert_array_equal(restored.steps, trajectory.steps)
    np.testing.assert_array_equal(
        restored["U"].values,
        trajectory["U"].values,
    )


def test_trajectory_requires_strictly_increasing_times_and_matching_frames() -> None:
    trajectory = _trajectory()
    layout = trajectory.layout
    field = TrajectoryField(FieldSpec("phi"), layout, np.ones((2, len(layout))))

    with pytest.raises(ValueError, match="strictly increasing"):
        Trajectory(layout, [0.0, 0.0], [field])
    with pytest.raises(ValueError, match="has 2 frames"):
        Trajectory(layout, [0.0], [field])


def test_trajectory_never_overwrites_and_detects_tampering(tmp_path) -> None:
    target = write_trajectory(tmp_path / "trajectory", _trajectory())

    with pytest.raises(FileExistsError, match="already exists"):
        write_trajectory(target, _trajectory())

    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["topology_id"] = "tampered"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_trajectory(target)
