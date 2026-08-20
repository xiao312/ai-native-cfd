"""Transparent, non-pickle checkpoints for small adaptive states."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from aicfd.fields.field import CellField, FieldSpec
from aicfd.fields.layout import TreeLayout
from aicfd.fields.state import State
from aicfd.representation import Cell

_SCHEMA_VERSION = 1


def write_checkpoint(path: str | Path, state: State) -> Path:
    """Write topology, field metadata, and arrays into a new directory.

    Existing paths are never overwritten. This keeps the first implementation
    deliberately safe and makes interrupted or accidental writes obvious.
    """

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"checkpoint path already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()

    levels = np.array([cell.level for cell in state.layout.cells], dtype=np.int64)
    indices = np.array([cell.index for cell in state.layout.cells], dtype=np.int64)
    np.savez(target / "topology.npz", levels=levels, indices=indices)

    arrays: dict[str, np.ndarray[Any, Any]] = {}
    field_metadata: list[dict[str, Any]] = []
    for number, field in enumerate(state.fields.values()):
        storage_key = f"field_{number}"
        arrays[storage_key] = field.values
        field_metadata.append(
            {
                "storage_key": storage_key,
                "spec": field.spec.to_dict(),
                "shape": list(field.values.shape),
            }
        )
    np.savez(target / "fields.npz", **arrays)

    metadata = {
        "schema_version": _SCHEMA_VERSION,
        "dimension": state.layout.dimension,
        "origin": list(state.layout.origin),
        "extent": list(state.layout.extent),
        "topology_id": state.layout.topology_id,
        "time": state.time,
        "step": state.step,
        "fields": field_metadata,
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def _load_metadata(path: Path) -> dict[str, Any]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a JSON object")
    if metadata.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    return metadata


def load_checkpoint(path: str | Path) -> State:
    """Restore and validate a checkpoint produced by :func:`write_checkpoint`."""

    source = Path(path)
    metadata = _load_metadata(source / "metadata.json")
    with np.load(source / "topology.npz", allow_pickle=False) as topology:
        levels = np.array(topology["levels"], dtype=np.int64, copy=True)
        indices = np.array(topology["indices"], dtype=np.int64, copy=True)

    dimension = int(metadata["dimension"])
    if levels.ndim != 1 or indices.shape != (len(levels), dimension):
        raise ValueError("checkpoint topology arrays have inconsistent shapes")
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
        raise ValueError("checkpoint topology fingerprint does not match its data")

    fields: list[CellField] = []
    with np.load(source / "fields.npz", allow_pickle=False) as arrays:
        for item in metadata["fields"]:
            spec = FieldSpec.from_dict(item["spec"])
            key = item["storage_key"]
            if key not in arrays:
                raise ValueError(f"checkpoint is missing array {key!r}")
            values = np.array(arrays[key], copy=True)
            if list(values.shape) != item["shape"]:
                raise ValueError(
                    f"checkpoint shape metadata is wrong for {spec.name!r}"
                )
            fields.append(CellField(spec, layout, values))

    return State(
        layout,
        fields,
        time=float(metadata["time"]),
        step=int(metadata["step"]),
    )
