"""Fixed-topology time trajectories for small reference simulations."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aicfd.fields.field import (
    CellField,
    FieldLocation,
    FieldSpec,
    ValueRepresentation,
)
from aicfd.fields.layout import TreeLayout
from aicfd.fields.state import State
from aicfd.representation import Cell

_SCHEMA_VERSION = 1


class TrajectoryField:
    """Values of one physical field at every time on one fixed layout."""

    __slots__ = ("layout", "spec", "values")

    def __init__(
        self,
        spec: FieldSpec,
        layout: TreeLayout,
        values: ArrayLike,
    ) -> None:
        if spec.location is not FieldLocation.CELL:
            raise ValueError("TrajectoryField currently stores only cell fields")
        if spec.value_representation not in (
            ValueRepresentation.CELL_AVERAGE,
            ValueRepresentation.CELL_INTEGRAL,
            ValueRepresentation.POINT_VALUE,
        ):
            raise ValueError(
                "TrajectoryField requires a cell-compatible representation"
            )
        array = np.array(values, dtype=spec.dtype, copy=True)
        trailing_shape = (
            (len(layout),)
            if spec.n_components == 1
            else (len(layout), spec.n_components)
        )
        if array.ndim != len(trailing_shape) + 1 or array.shape[1:] != trailing_shape:
            raise ValueError(
                f"trajectory field {spec.name!r} needs shape "
                f"(n_times, {', '.join(map(str, trailing_shape))}), got {array.shape}"
            )
        if array.shape[0] == 0:
            raise ValueError("trajectory fields need at least one time frame")
        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"trajectory field {spec.name!r} contains non-finite values"
            )
        if spec.lower_bound is not None and np.any(array < spec.lower_bound):
            raise ValueError(f"trajectory field {spec.name!r} is below its lower bound")
        if spec.upper_bound is not None and np.any(array > spec.upper_bound):
            raise ValueError(f"trajectory field {spec.name!r} is above its upper bound")

        self.spec = spec
        self.layout = layout
        self.values = array

    @property
    def n_times(self) -> int:
        """Number of stored time frames."""

        return int(self.values.shape[0])

    def frame(self, index: int) -> CellField:
        """Return one time frame as an independent :class:`CellField`."""

        return CellField(self.spec, self.layout, self.values[index])


class Trajectory:
    """A sequence of states that all share one immutable tree layout.

    This deliberately small first representation suits a fixed OpenFOAM mesh.
    A later adaptive trajectory may store a different ``State`` per frame, but
    keeping topology fixed first lets us validate field order and time handling
    independently from remeshing.
    """

    __slots__ = (
        "_attributes",
        "_attributes_view",
        "_fields",
        "_fields_view",
        "layout",
        "steps",
        "times",
    )

    def __init__(
        self,
        layout: TreeLayout,
        times: ArrayLike,
        fields: Iterable[TrajectoryField] | Mapping[str, TrajectoryField],
        *,
        steps: ArrayLike | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        time_array = np.array(times, dtype=np.float64, copy=True)
        if time_array.ndim != 1 or len(time_array) == 0:
            raise ValueError(
                "trajectory times must be a non-empty one-dimensional array"
            )
        if not np.all(np.isfinite(time_array)):
            raise ValueError("trajectory times must be finite")
        if len(time_array) > 1 and np.any(np.diff(time_array) <= 0.0):
            raise ValueError("trajectory times must be strictly increasing")

        if steps is None:
            step_array = np.arange(len(time_array), dtype=np.int64)
        else:
            raw_steps = np.asarray(steps)
            if raw_steps.dtype.kind not in "iu":
                raise TypeError("trajectory steps must contain integers")
            step_array = np.array(raw_steps, dtype=np.int64, copy=True)
        if step_array.shape != time_array.shape:
            raise ValueError("trajectory steps and times must have the same shape")
        if np.any(step_array < 0):
            raise ValueError("trajectory steps must be non-negative")
        if len(step_array) > 1 and np.any(np.diff(step_array) <= 0):
            raise ValueError("trajectory steps must be strictly increasing")

        candidates = (
            tuple(fields.items())
            if isinstance(fields, Mapping)
            else tuple((field.spec.name, field) for field in fields)
        )
        registry: dict[str, TrajectoryField] = {}
        for name, field in candidates:
            if not isinstance(field, TrajectoryField):
                raise TypeError("trajectory fields must be TrajectoryField instances")
            if name != field.spec.name:
                raise ValueError("field registry key must match FieldSpec.name")
            if name in registry:
                raise ValueError(f"duplicate trajectory field name {name!r}")
            if field.layout.topology_id != layout.topology_id:
                raise ValueError(
                    f"trajectory field {name!r} belongs to a different layout"
                )
            if field.n_times != len(time_array):
                raise ValueError(
                    f"trajectory field {name!r} has {field.n_times} frames, "
                    f"expected {len(time_array)}"
                )
            registry[name] = field
        if not registry:
            raise ValueError("a trajectory needs at least one physical field")

        attribute_data = json.loads(json.dumps(dict(attributes or {})))
        if not isinstance(attribute_data, dict):
            raise TypeError("trajectory attributes must form a JSON object")

        self.layout = layout
        self.times = time_array
        self.steps = step_array
        self._fields = registry
        self._fields_view = MappingProxyType(registry)
        self._attributes = attribute_data
        self._attributes_view = MappingProxyType(attribute_data)

    def __len__(self) -> int:
        return len(self.times)

    @property
    def fields(self) -> Mapping[str, TrajectoryField]:
        """Read-only field registry keyed by field name."""

        return self._fields_view

    @property
    def attributes(self) -> Mapping[str, Any]:
        """Read-only JSON-compatible source metadata."""

        return self._attributes_view

    def __contains__(self, field_name: object) -> bool:
        return field_name in self._fields

    def __getitem__(self, field_name: str) -> TrajectoryField:
        try:
            return self._fields[field_name]
        except KeyError as error:
            raise KeyError(f"trajectory has no field named {field_name!r}") from error

    def state(self, index: int) -> State:
        """Materialize one frame using the existing single-state abstraction."""

        normalized_index = range(len(self))[index]
        return State(
            self.layout,
            [field.frame(normalized_index) for field in self._fields.values()],
            time=float(self.times[normalized_index]),
            step=int(self.steps[normalized_index]),
        )


def write_trajectory(path: str | Path, trajectory: Trajectory) -> Path:
    """Write one fixed-layout trajectory without pickle or path overwrites."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"trajectory path already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()

    levels = np.array([cell.level for cell in trajectory.layout.cells], dtype=np.int64)
    indices = np.array([cell.index for cell in trajectory.layout.cells], dtype=np.int64)
    np.savez(target / "topology.npz", levels=levels, indices=indices)

    arrays: dict[str, NDArray[Any]] = {
        "times": trajectory.times,
        "steps": trajectory.steps,
    }
    field_metadata: list[dict[str, Any]] = []
    for number, field in enumerate(trajectory.fields.values()):
        storage_key = f"field_{number}"
        arrays[storage_key] = field.values
        field_metadata.append(
            {
                "storage_key": storage_key,
                "spec": field.spec.to_dict(),
                "shape": list(field.values.shape),
            }
        )
    np.savez_compressed(target / "trajectory.npz", **arrays)

    metadata = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "fixed_layout_trajectory",
        "dimension": trajectory.layout.dimension,
        "origin": list(trajectory.layout.origin),
        "extent": list(trajectory.layout.extent),
        "topology_id": trajectory.layout.topology_id,
        "n_times": len(trajectory),
        "fields": field_metadata,
        "attributes": dict(trajectory.attributes),
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def load_trajectory(path: str | Path) -> Trajectory:
    """Restore and validate a trajectory produced by :func:`write_trajectory`."""

    source = Path(path)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("trajectory metadata must be a JSON object")
    if metadata.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported trajectory schema version")
    if metadata.get("kind") != "fixed_layout_trajectory":
        raise ValueError("unsupported trajectory kind")

    with np.load(source / "topology.npz", allow_pickle=False) as topology:
        levels = np.array(topology["levels"], dtype=np.int64, copy=True)
        indices = np.array(topology["indices"], dtype=np.int64, copy=True)

    dimension = int(metadata["dimension"])
    if levels.ndim != 1 or indices.shape != (len(levels), dimension):
        raise ValueError("trajectory topology arrays have inconsistent shapes")
    cells = tuple(
        Cell(level=int(level), index=tuple(int(value) for value in index))
        for level, index in zip(levels, indices, strict=True)
    )
    layout = TreeLayout(
        dimension=dimension,
        origin=tuple(metadata["origin"]),
        extent=tuple(metadata["extent"]),
        cells=cells,
    )
    if layout.topology_id != metadata["topology_id"]:
        raise ValueError("trajectory topology fingerprint does not match its data")

    fields: list[TrajectoryField] = []
    with np.load(source / "trajectory.npz", allow_pickle=False) as arrays:
        times = np.array(arrays["times"], dtype=np.float64, copy=True)
        steps = np.array(arrays["steps"], dtype=np.int64, copy=True)
        for item in metadata["fields"]:
            spec = FieldSpec.from_dict(item["spec"])
            key = item["storage_key"]
            if key not in arrays:
                raise ValueError(f"trajectory is missing array {key!r}")
            values = np.array(arrays[key], copy=True)
            if list(values.shape) != item["shape"]:
                raise ValueError(
                    f"trajectory shape metadata is wrong for {spec.name!r}"
                )
            fields.append(TrajectoryField(spec, layout, values))

    if len(times) != int(metadata["n_times"]):
        raise ValueError("trajectory time count does not match its metadata")
    return Trajectory(
        layout,
        times,
        fields,
        steps=steps,
        attributes=metadata.get("attributes", {}),
    )
