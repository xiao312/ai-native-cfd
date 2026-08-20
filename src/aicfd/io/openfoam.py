"""Small OpenFOAM ASCII readers for fixed Cartesian reference cases.

The parser intentionally reads only the ``internalField`` of scalar and vector
volume fields. It is not a replacement for OpenFOAM's own I/O library; it is a
transparent adapter for the tiny, controlled tutorial cases used by this project.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from math import isfinite
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aicfd.fields import FieldSpec, Trajectory, TrajectoryField, TreeLayout
from aicfd.representation import AdaptiveTree, Cell

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_HEADER_CLASS = re.compile(r"\bclass\s+([^;\s]+)\s*;")
_HEADER_OBJECT = re.compile(r"\bobject\s+([^;\s]+)\s*;")
_DIMENSIONS = re.compile(r"\bdimensions\s*\[([^]]+)]\s*;")
_UNIFORM_FIELD = re.compile(r"\binternalField\s+uniform\s+([^;]+)\s*;")
_NONUNIFORM_FIELD = re.compile(
    r"\binternalField\s+nonuniform\s+List<([^>]+)>\s+"
    r"(\d+)\s*\((.*?)\)\s*;",
    re.DOTALL,
)
_VECTOR_TOKEN = re.compile(r"\(([^()]*)\)")


@dataclass(frozen=True, slots=True)
class FoamField:
    """Parsed metadata and internal values from one OpenFOAM field file."""

    name: str
    field_class: str
    dimensions: tuple[float, ...]
    values: NDArray[np.float64]
    uniform: bool

    def values_for_count(self, count: int) -> NDArray[np.float64]:
        """Expand a uniform value or validate a nonuniform cell list."""

        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("field expansion count must be a positive integer")
        if self.uniform:
            return np.repeat(self.values, count, axis=0)
        if len(self.values) != count:
            raise ValueError(
                f"OpenFOAM field {self.name!r} contains {len(self.values)} values, "
                f"expected {count}"
            )
        return self.values.copy()


@dataclass(frozen=True, slots=True)
class OpenFoamImportReport:
    """Checks performed while mapping one Cartesian trajectory."""

    source_cells: int
    time_frames: int
    first_time: float
    last_time: float
    maximum_center_error: float
    maximum_out_of_plane_velocity: float


def _clean_foam_text(text: str) -> str:
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", text))


def _parse_vector(token: str, *, context: str) -> NDArray[np.float64]:
    values = np.fromstring(token.strip().strip("()"), sep=" ", dtype=np.float64)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"could not parse finite values for {context}")
    return values


def read_foam_field(path: str | Path) -> FoamField:
    """Read an ASCII scalar or vector ``internalField``.

    Uniform fields are retained as a one-row array and can later be expanded with
    :meth:`FoamField.values_for_count` once the mesh cell count is known.
    """

    source = Path(path)
    text = _clean_foam_text(source.read_text(encoding="utf-8"))
    if not re.search(r"\bformat\s+ascii\s*;", text):
        raise ValueError(f"OpenFOAM field must use ASCII format: {source}")

    class_match = _HEADER_CLASS.search(text)
    object_match = _HEADER_OBJECT.search(text)
    dimensions_match = _DIMENSIONS.search(text)
    if class_match is None or object_match is None or dimensions_match is None:
        raise ValueError(f"OpenFOAM field header is incomplete: {source}")
    dimensions = tuple(float(value) for value in dimensions_match.group(1).split())
    if len(dimensions) != 7 or not all(isfinite(value) for value in dimensions):
        raise ValueError(f"OpenFOAM dimensions must contain seven powers: {source}")

    field_class = class_match.group(1)
    is_vector = field_class.endswith("VectorField")
    is_scalar = field_class.endswith("ScalarField")
    if not (is_scalar or is_vector):
        raise ValueError(f"unsupported OpenFOAM field class {field_class!r}")

    nonuniform_match = _NONUNIFORM_FIELD.search(text)
    if nonuniform_match is not None:
        list_type, count_text, body = nonuniform_match.groups()
        count = int(count_text)
        expected_type = "vector" if is_vector else "scalar"
        if list_type != expected_type:
            raise ValueError(
                f"field class {field_class!r} conflicts with List<{list_type}>"
            )
        if is_vector:
            rows = [
                _parse_vector(token, context=f"vector in {source}")
                for token in _VECTOR_TOKEN.findall(body)
            ]
            if len(rows) != count or any(len(row) != 3 for row in rows):
                raise ValueError(f"OpenFOAM vector list count is wrong: {source}")
            values = np.stack(rows)
        else:
            values = np.fromstring(body, sep=" ", dtype=np.float64)
            if values.shape != (count,):
                raise ValueError(f"OpenFOAM scalar list count is wrong: {source}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"OpenFOAM field contains non-finite values: {source}")
        return FoamField(
            name=object_match.group(1),
            field_class=field_class,
            dimensions=dimensions,
            values=values,
            uniform=False,
        )

    uniform_match = _UNIFORM_FIELD.search(text)
    if uniform_match is None:
        raise ValueError(f"OpenFOAM field has no supported internalField: {source}")
    row = _parse_vector(uniform_match.group(1), context=f"uniform value in {source}")
    expected_components = 3 if is_vector else 1
    if len(row) != expected_components:
        raise ValueError(
            f"uniform {field_class} needs {expected_components} components: {source}"
        )
    values = row.reshape(1, expected_components) if is_vector else row
    return FoamField(
        name=object_match.group(1),
        field_class=field_class,
        dimensions=dimensions,
        values=values,
        uniform=True,
    )


def discover_time_directories(
    case_path: str | Path,
    *,
    required_fields: tuple[str, ...] = ("U", "p"),
) -> tuple[tuple[Decimal, Path], ...]:
    """Find numeric time directories that contain every requested field."""

    case = Path(case_path)
    discovered: list[tuple[Decimal, Path]] = []
    for candidate in case.iterdir():
        if not candidate.is_dir():
            continue
        try:
            time = Decimal(candidate.name)
        except InvalidOperation:
            continue
        if not time.is_finite():
            continue
        if all((candidate / field_name).is_file() for field_name in required_fields):
            discovered.append((time, candidate))
    discovered.sort(key=lambda item: item[0])
    if not discovered:
        raise ValueError(f"no complete OpenFOAM time directories found in {case}")
    return tuple(discovered)


def _uniform_tree(
    level: int,
    origin: tuple[float, float],
    extent: tuple[float, float],
) -> AdaptiveTree:
    tree = AdaptiveTree(2, origin=origin, extent=extent)
    for _ in range(level):
        tree.refine_many(tree.leaves)
    return tree


def _read_cell_centers(case: Path, count: int) -> NDArray[np.float64]:
    vector_path = case / "0" / "C"
    if vector_path.is_file():
        centers = read_foam_field(vector_path).values_for_count(count)
        if centers.shape != (count, 3):
            raise ValueError("OpenFOAM cell-centre field C must be a vector field")
        return centers

    component_paths = tuple(case / "0" / name for name in ("Cx", "Cy", "Cz"))
    if all(path.is_file() for path in component_paths):
        components = [
            read_foam_field(path).values_for_count(count) for path in component_paths
        ]
        if any(values.shape != (count,) for values in components):
            raise ValueError("OpenFOAM cell-centre components must be scalar fields")
        return np.column_stack(components)
    raise FileNotFoundError(
        "cell centres are missing; run `postProcess -func writeCellCentres -time 0`"
    )


def _cartesian_row_mapping(
    centers: NDArray[np.float64],
    layout: TreeLayout,
    level: int,
    tolerance: float,
) -> tuple[NDArray[np.int64], float]:
    cells_per_axis = 1 << level
    coordinates = centers[:, :2]
    origin = np.asarray(layout.origin)
    extent = np.asarray(layout.extent)
    scaled = (coordinates - origin) * cells_per_axis / extent
    cell_indices = np.floor(scaled).astype(np.int64)
    if np.any(cell_indices < 0) or np.any(cell_indices >= cells_per_axis):
        raise ValueError("an OpenFOAM cell centre lies outside the requested domain")

    expected_centers = origin + (cell_indices + 0.5) * extent / cells_per_axis
    errors = np.linalg.norm(coordinates - expected_centers, axis=1)
    maximum_error = float(np.max(errors, initial=0.0))
    if maximum_error > tolerance:
        raise ValueError(
            f"OpenFOAM mesh is not the requested Cartesian level-{level} grid; "
            f"maximum centre error is {maximum_error:.6g}"
        )

    target_rows = np.array(
        [
            layout.index(Cell(level, tuple(int(value) for value in index)))
            for index in cell_indices
        ],
        dtype=np.int64,
    )
    if len(np.unique(target_rows)) != len(target_rows):
        raise ValueError("multiple OpenFOAM cells map to the same quadtree leaf")
    return target_rows, maximum_error


def import_openfoam_cartesian_2d(
    case_path: str | Path,
    *,
    level: int,
    origin: tuple[float, float],
    extent: tuple[float, float],
    delta_t: float | None = None,
    openfoam_version: str | None = None,
    coordinate_tolerance: float | None = None,
    out_of_plane_tolerance: float = 1.0e-12,
) -> tuple[Trajectory, OpenFoamImportReport]:
    """Import ``U`` and ``p`` from a fixed 2D Cartesian OpenFOAM trajectory."""

    if not isinstance(level, int) or isinstance(level, bool) or level < 0:
        raise ValueError("quadtree level must be a non-negative integer")
    expected_cells = (1 << level) ** 2
    tree = _uniform_tree(level, origin, extent)
    layout = TreeLayout.from_tree(tree)
    case = Path(case_path)
    centers = _read_cell_centers(case, expected_cells)
    tolerance = (
        max(extent) * 1.0e-8
        if coordinate_tolerance is None
        else float(coordinate_tolerance)
    )
    if tolerance < 0.0 or not isfinite(tolerance):
        raise ValueError("coordinate tolerance must be finite and non-negative")
    source_to_target, maximum_center_error = _cartesian_row_mapping(
        centers,
        layout,
        level,
        tolerance,
    )

    discovered = discover_time_directories(case)
    times = np.array([float(time) for time, _ in discovered], dtype=np.float64)
    if delta_t is None:
        steps = np.arange(len(times), dtype=np.int64)
    else:
        delta_t = float(delta_t)
        if delta_t <= 0.0 or not isfinite(delta_t):
            raise ValueError("delta_t must be finite and positive")
        raw_steps = (times - times[0]) / delta_t
        steps = np.rint(raw_steps).astype(np.int64)
        if not np.allclose(raw_steps, steps, rtol=0.0, atol=1.0e-7):
            raise ValueError("OpenFOAM time names are inconsistent with delta_t")

    velocity_frames = np.empty((len(times), expected_cells, 2), dtype=np.float64)
    pressure_frames = np.empty((len(times), expected_cells), dtype=np.float64)
    maximum_out_of_plane_velocity = 0.0
    for frame, (_, time_directory) in enumerate(discovered):
        velocity = read_foam_field(time_directory / "U")
        pressure = read_foam_field(time_directory / "p")
        if velocity.dimensions != (0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0):
            raise ValueError(
                f"unexpected velocity dimensions in {time_directory / 'U'}"
            )
        if pressure.dimensions != (0.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0):
            raise ValueError(
                f"unexpected pressure dimensions in {time_directory / 'p'}"
            )
        source_velocity = velocity.values_for_count(expected_cells)
        source_pressure = pressure.values_for_count(expected_cells)
        if source_velocity.shape != (expected_cells, 3):
            raise ValueError("OpenFOAM U must contain three vector components")
        if source_pressure.shape != (expected_cells,):
            raise ValueError("OpenFOAM p must be a scalar cell field")
        out_of_plane = float(np.max(np.abs(source_velocity[:, 2]), initial=0.0))
        maximum_out_of_plane_velocity = max(
            maximum_out_of_plane_velocity,
            out_of_plane,
        )
        if out_of_plane > out_of_plane_tolerance:
            raise ValueError(
                f"OpenFOAM U is not two-dimensional at time {time_directory.name}"
            )

        velocity_frames[frame, source_to_target] = source_velocity[:, :2]
        pressure_frames[frame, source_to_target] = source_pressure

    velocity_spec = FieldSpec(
        "U",
        component_names=("x", "y"),
        unit="m/s",
        unit_dimensions=(0, 1, -1, 0, 0, 0, 0),
    )
    pressure_spec = FieldSpec(
        "p",
        unit="m^2/s^2",
        unit_dimensions=(0, 2, -2, 0, 0, 0, 0),
    )
    attributes = {
        "source": "OpenFOAM",
        "openfoam_version": openfoam_version,
        "case_name": case.name,
        "source_time_names": [directory.name for _, directory in discovered],
        "mapping": "exact Cartesian cell-centre mapping to quadtree Morton order",
        "pressure_kind": "kinematic pressure with an arbitrary reference offset",
    }
    trajectory = Trajectory(
        layout,
        times,
        [
            TrajectoryField(velocity_spec, layout, velocity_frames),
            TrajectoryField(pressure_spec, layout, pressure_frames),
        ],
        steps=steps,
        attributes=attributes,
    )
    report = OpenFoamImportReport(
        source_cells=expected_cells,
        time_frames=len(times),
        first_time=float(times[0]),
        last_time=float(times[-1]),
        maximum_center_error=maximum_center_error,
        maximum_out_of_plane_velocity=maximum_out_of_plane_velocity,
    )
    return trajectory, report
