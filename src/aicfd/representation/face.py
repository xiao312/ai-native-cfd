"""Explicit axis-aligned face segments for adaptive Cartesian leaves."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from math import isclose, isfinite, prod
from types import MappingProxyType
from typing import TYPE_CHECKING

from aicfd.representation.cell import Cell

if TYPE_CHECKING:
    from collections.abc import Mapping

    from aicfd.fields.layout import TreeLayout


class FaceRelation(str, Enum):
    """Topological relationship on the two sides of a face segment."""

    BOUNDARY = "boundary"
    SAME_LEVEL = "same_level"
    COARSE_FINE = "coarse_fine"


@dataclass(frozen=True, order=True, slots=True)
class FaceKey:
    """Stable dyadic identity of one non-overlapping face segment.

    ``index[axis]`` identifies a grid plane and may therefore equal ``2**level``.
    Every tangential index identifies an interval and is strictly smaller.
    """

    axis: int
    level: int
    index: tuple[int, ...]

    def __post_init__(self) -> None:
        index = tuple(self.index)
        object.__setattr__(self, "index", index)
        if not 1 <= len(index) <= 3:
            raise ValueError("face dimension must be 1, 2, or 3")
        if not isinstance(self.axis, int) or isinstance(self.axis, bool):
            raise TypeError("face axis must be an integer")
        if not 0 <= self.axis < len(index):
            raise ValueError("face axis is outside its dimension")
        if not isinstance(self.level, int) or isinstance(self.level, bool):
            raise TypeError("face level must be an integer")
        if self.level < 0:
            raise ValueError("face level must be non-negative")

        cells_per_axis = 1 << self.level
        for axis, coordinate in enumerate(index):
            if not isinstance(coordinate, int) or isinstance(coordinate, bool):
                raise TypeError("face indices must be integers")
            upper = cells_per_axis if axis == self.axis else cells_per_axis - 1
            if not 0 <= coordinate <= upper:
                raise ValueError("face index is outside its dyadic level")

    @property
    def dimension(self) -> int:
        return len(self.index)

    @property
    def stable_id(self) -> str:
        coordinates = ",".join(str(value) for value in self.index)
        return f"{self.dimension}D:F:L{self.level}:A{self.axis}:I{coordinates}"


@dataclass(frozen=True, slots=True)
class FaceSegment:
    """One oriented finite-volume interaction surface.

    For an interior face, the normal points from ``owner`` to ``neighbor``. The
    owner is therefore always on the geometrically lower side of the face. For a
    boundary face, the normal points outward from the only adjacent cell.
    """

    key: FaceKey
    owner: Cell
    neighbor: Cell | None
    bounds: tuple[tuple[float, float], ...]
    center: tuple[float, ...]
    normal: tuple[int, ...]
    area: float

    def __post_init__(self) -> None:
        bounds = tuple(tuple(float(value) for value in pair) for pair in self.bounds)
        center = tuple(float(value) for value in self.center)
        normal = tuple(self.normal)
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "normal", normal)

        dimension = self.key.dimension
        if self.owner.dimension != dimension:
            raise ValueError("face owner dimension does not match its key")
        if self.neighbor is not None and (
            self.neighbor.dimension != dimension or self.neighbor == self.owner
        ):
            raise ValueError("face neighbor must be a distinct compatible cell")
        if len(bounds) != dimension or len(center) != dimension:
            raise ValueError("face geometry dimension is inconsistent")
        if any(
            len(pair) != 2
            or not all(isfinite(value) for value in pair)
            or pair[0] > pair[1]
            for pair in bounds
        ):
            raise ValueError("face bounds must contain ordered finite pairs")
        if not all(isfinite(value) for value in center):
            raise ValueError("face center must be finite")
        if len(normal) != dimension or any(value not in (-1, 0, 1) for value in normal):
            raise ValueError("face normal must contain only -1, 0, or 1")
        if sum(value != 0 for value in normal) != 1 or normal[self.key.axis] == 0:
            raise ValueError("face normal must be a signed basis vector on its axis")
        if not isfinite(self.area) or self.area <= 0.0:
            raise ValueError("face area must be finite and positive")

    @property
    def relation(self) -> FaceRelation:
        if self.neighbor is None:
            return FaceRelation.BOUNDARY
        if self.owner.level == self.neighbor.level:
            return FaceRelation.SAME_LEVEL
        return FaceRelation.COARSE_FINE

    @property
    def coarse_cell(self) -> Cell | None:
        if self.relation is not FaceRelation.COARSE_FINE:
            return None
        assert self.neighbor is not None
        return min((self.owner, self.neighbor), key=lambda cell: cell.level)

    @property
    def fine_cell(self) -> Cell | None:
        if self.relation is not FaceRelation.COARSE_FINE:
            return None
        assert self.neighbor is not None
        return max((self.owner, self.neighbor), key=lambda cell: cell.level)

    def incidence_sign(self, cell: Cell) -> int:
        """Return the sign converting face orientation to cell-outward orientation."""

        if cell == self.owner:
            return 1
        if cell == self.neighbor:
            return -1
        raise KeyError(f"{cell} is not incident to face {self.key.stable_id}")

    def other_cell(self, cell: Cell) -> Cell | None:
        if cell == self.owner:
            return self.neighbor
        if cell == self.neighbor:
            return self.owner
        raise KeyError(f"{cell} is not incident to face {self.key.stable_id}")


@dataclass(frozen=True, slots=True)
class FaceIncidence:
    """A face row and its orientation relative to one cell."""

    face_index: int
    sign: int

    def __post_init__(self) -> None:
        if self.face_index < 0:
            raise ValueError("face incidence index must be non-negative")
        if self.sign not in (-1, 1):
            raise ValueError("face incidence sign must be -1 or 1")


def _integer_bounds(cell: Cell, common_level: int) -> tuple[tuple[int, int], ...]:
    scale = 1 << (common_level - cell.level)
    return tuple(
        (coordinate * scale, (coordinate + 1) * scale) for coordinate in cell.index
    )


def _physical_face_geometry(
    layout: TreeLayout,
    key: FaceKey,
    integer_bounds: tuple[tuple[int, int], ...],
) -> tuple[
    tuple[tuple[float, float], ...],
    tuple[float, ...],
    float,
]:
    scale = 1 << key.level
    bounds = tuple(
        (
            origin + extent * lower / scale,
            origin + extent * upper / scale,
        )
        for origin, extent, (lower, upper) in zip(
            layout.origin,
            layout.extent,
            integer_bounds,
            strict=True,
        )
    )
    center = tuple(0.5 * (lower + upper) for lower, upper in bounds)
    tangential_lengths = (
        upper - lower for axis, (lower, upper) in enumerate(bounds) if axis != key.axis
    )
    return bounds, center, prod(tangential_lengths)


def _shared_face(
    layout: TreeLayout,
    first: Cell,
    second: Cell,
) -> FaceSegment | None:
    common_level = max(first.level, second.level)
    first_bounds = _integer_bounds(first, common_level)
    second_bounds = _integer_bounds(second, common_level)
    touching: tuple[int, Cell, Cell, int] | None = None
    overlap_bounds: list[tuple[int, int]] = []

    for axis, ((first_lower, first_upper), (second_lower, second_upper)) in enumerate(
        zip(first_bounds, second_bounds, strict=True)
    ):
        if first_upper == second_lower:
            if touching is not None:
                return None
            touching = (axis, first, second, first_upper)
            overlap_bounds.append((first_upper, first_upper))
            continue
        if second_upper == first_lower:
            if touching is not None:
                return None
            touching = (axis, second, first, second_upper)
            overlap_bounds.append((second_upper, second_upper))
            continue
        lower = max(first_lower, second_lower)
        upper = min(first_upper, second_upper)
        if upper <= lower:
            return None
        overlap_bounds.append((lower, upper))

    if touching is None:
        return None
    axis, owner, neighbor, plane = touching
    if any(
        upper - lower != 1
        for tangent_axis, (lower, upper) in enumerate(overlap_bounds)
        if tangent_axis != axis
    ):
        raise RuntimeError("adaptive face segment is not one dyadic face tile")
    key_index = tuple(
        plane if coordinate_axis == axis else overlap_bounds[coordinate_axis][0]
        for coordinate_axis in range(layout.dimension)
    )
    key = FaceKey(axis=axis, level=common_level, index=key_index)
    bounds, center, area = _physical_face_geometry(
        layout,
        key,
        tuple(overlap_bounds),
    )
    normal = tuple(
        1 if coordinate_axis == axis else 0
        for coordinate_axis in range(layout.dimension)
    )
    return FaceSegment(key, owner, neighbor, bounds, center, normal, area)


def _boundary_face(
    layout: TreeLayout,
    cell: Cell,
    axis: int,
    side: int,
) -> FaceSegment:
    cells_per_axis = 1 << cell.level
    plane = cell.index[axis] if side < 0 else cell.index[axis] + 1
    key_index = tuple(
        plane if coordinate_axis == axis else cell.index[coordinate_axis]
        for coordinate_axis in range(layout.dimension)
    )
    key = FaceKey(axis=axis, level=cell.level, index=key_index)
    integer_bounds = tuple(
        (plane, plane)
        if coordinate_axis == axis
        else (cell.index[coordinate_axis], cell.index[coordinate_axis] + 1)
        for coordinate_axis in range(layout.dimension)
    )
    bounds, center, area = _physical_face_geometry(layout, key, integer_bounds)
    normal = tuple(
        side if coordinate_axis == axis else 0
        for coordinate_axis in range(layout.dimension)
    )
    if plane not in (0, cells_per_axis):
        raise RuntimeError("requested face is not on the physical domain boundary")
    return FaceSegment(key, cell, None, bounds, center, normal, area)


class FaceTopology:
    """Deterministic rows for all interior and domain-boundary face segments."""

    __slots__ = (
        "_cell_incidences",
        "_cell_incidences_view",
        "_key_to_index",
        "_key_to_index_view",
        "faces",
        "layout",
        "topology_id",
    )

    def __init__(self, layout: TreeLayout) -> None:
        faces_by_key: dict[FaceKey, FaceSegment] = {}

        for position, first in enumerate(layout.cells):
            for second in layout.cells[position + 1 :]:
                face = _shared_face(layout, first, second)
                if face is not None:
                    if face.key in faces_by_key:
                        raise RuntimeError("duplicate interior face segment")
                    faces_by_key[face.key] = face

        for cell in layout.cells:
            cells_per_axis = 1 << cell.level
            for axis in range(layout.dimension):
                if cell.index[axis] == 0:
                    face = _boundary_face(layout, cell, axis, -1)
                    if face.key in faces_by_key:
                        raise RuntimeError("duplicate lower boundary face segment")
                    faces_by_key[face.key] = face
                if cell.index[axis] + 1 == cells_per_axis:
                    face = _boundary_face(layout, cell, axis, 1)
                    if face.key in faces_by_key:
                        raise RuntimeError("duplicate upper boundary face segment")
                    faces_by_key[face.key] = face

        faces = tuple(faces_by_key[key] for key in sorted(faces_by_key))
        key_to_index = {face.key: index for index, face in enumerate(faces)}
        incidences: dict[Cell, list[FaceIncidence]] = {
            cell: [] for cell in layout.cells
        }
        for face_index, face in enumerate(faces):
            incidences[face.owner].append(FaceIncidence(face_index, 1))
            if face.neighbor is not None:
                incidences[face.neighbor].append(FaceIncidence(face_index, -1))

        self.layout = layout
        self.faces = faces
        self._key_to_index = key_to_index
        self._key_to_index_view = MappingProxyType(key_to_index)
        frozen_incidences = {
            cell: tuple(cell_incidences) for cell, cell_incidences in incidences.items()
        }
        self._cell_incidences = frozen_incidences
        self._cell_incidences_view = MappingProxyType(frozen_incidences)
        self.topology_id = self._fingerprint()
        self.assert_valid()

    def __len__(self) -> int:
        return len(self.faces)

    @property
    def key_to_index(self) -> Mapping[FaceKey, int]:
        return self._key_to_index_view

    @property
    def cell_incidences(self) -> Mapping[Cell, tuple[FaceIncidence, ...]]:
        return self._cell_incidences_view

    @property
    def boundary_faces(self) -> tuple[FaceSegment, ...]:
        return tuple(face for face in self.faces if face.neighbor is None)

    @property
    def interior_faces(self) -> tuple[FaceSegment, ...]:
        return tuple(face for face in self.faces if face.neighbor is not None)

    @property
    def coarse_fine_faces(self) -> tuple[FaceSegment, ...]:
        return tuple(
            face for face in self.faces if face.relation is FaceRelation.COARSE_FINE
        )

    def index(self, face: FaceKey | FaceSegment) -> int:
        key = face.key if isinstance(face, FaceSegment) else face
        try:
            return self._key_to_index[key]
        except KeyError as error:
            raise KeyError(f"{key} is not a face in this topology") from error

    def incidences(self, cell: Cell) -> tuple[FaceIncidence, ...]:
        try:
            return self._cell_incidences[cell]
        except KeyError as error:
            raise KeyError(f"{cell} is not active in this face topology") from error

    def cell_side_faces(
        self,
        cell: Cell,
        axis: int,
        side: int,
    ) -> tuple[FaceSegment, ...]:
        """Return all segments covering one outward side of an active cell."""

        if not isinstance(axis, int) or isinstance(axis, bool):
            raise TypeError("face axis must be an integer")
        if not 0 <= axis < self.layout.dimension:
            raise ValueError("face axis is outside the topology dimension")
        if side not in (-1, 1):
            raise ValueError("cell side must be -1 or 1")
        return tuple(
            self.faces[incidence.face_index]
            for incidence in self.incidences(cell)
            if incidence.sign * self.faces[incidence.face_index].normal[axis] == side
        )

    def assert_valid(self) -> None:
        """Check incidence and complete face coverage for every active cell."""

        active = set(self.layout.cells)
        for face in self.faces:
            if face.owner not in active or (
                face.neighbor is not None and face.neighbor not in active
            ):
                raise RuntimeError("a face references a non-active cell")

        for cell in self.layout.cells:
            cell_scale = 1 << cell.level
            for axis in range(self.layout.dimension):
                expected_area = prod(
                    extent / cell_scale
                    for coordinate_axis, extent in enumerate(self.layout.extent)
                    if coordinate_axis != axis
                )
                for side in (-1, 1):
                    actual_area = sum(
                        face.area for face in self.cell_side_faces(cell, axis, side)
                    )
                    if not isclose(
                        actual_area,
                        expected_area,
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-14,
                    ):
                        raise RuntimeError(
                            f"face segments do not cover {cell} axis {axis} side {side}"
                        )

    def _fingerprint(self) -> str:
        payload = {
            "layout": self.layout.topology_id,
            "faces": [
                {
                    "key": [face.key.axis, face.key.level, *face.key.index],
                    "owner": face.owner.stable_id,
                    "neighbor": (
                        None if face.neighbor is None else face.neighbor.stable_id
                    ),
                }
                for face in self.faces
            ],
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        return sha256(encoded).hexdigest()
