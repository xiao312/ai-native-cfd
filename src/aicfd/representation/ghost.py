"""Conceptual ghost slots and their value-source recipes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING

from aicfd.representation.cell import Cell
from aicfd.representation.face import FaceKey, FaceTopology

if TYPE_CHECKING:
    from collections.abc import Mapping


class GhostSourceKind(str, Enum):
    """How a later stencil filler should obtain one ghost value."""

    SAME_LEVEL = "same_level"
    COARSE_PROLONGATION = "coarse_prolongation"
    FINE_RESTRICTION = "fine_restriction"
    PHYSICAL_BOUNDARY = "physical_boundary"


@dataclass(frozen=True, slots=True)
class GhostSlot:
    """One cell-side stencil slot and the cells/faces that can fill it.

    A ghost slot is not an additional finite-volume cell. It is a recipe for a
    temporary stencil value beside one active cell.
    """

    cell: Cell
    axis: int
    side: int
    source_kind: GhostSourceKind
    source_cells: tuple[Cell, ...]
    face_keys: tuple[FaceKey, ...]

    def __post_init__(self) -> None:
        sources = tuple(self.source_cells)
        face_keys = tuple(self.face_keys)
        source_kind = GhostSourceKind(self.source_kind)
        object.__setattr__(self, "source_cells", sources)
        object.__setattr__(self, "face_keys", face_keys)
        object.__setattr__(self, "source_kind", source_kind)

        if not isinstance(self.axis, int) or isinstance(self.axis, bool):
            raise TypeError("ghost axis must be an integer")
        if not 0 <= self.axis < self.cell.dimension:
            raise ValueError("ghost axis is outside its cell dimension")
        if self.side not in (-1, 1):
            raise ValueError("ghost side must be -1 or 1")
        if not face_keys or any(
            key.dimension != self.cell.dimension or key.axis != self.axis
            for key in face_keys
        ):
            raise ValueError("ghost face keys must cover the requested cell side")
        if len(set(sources)) != len(sources):
            raise ValueError("ghost source cells must be unique")
        if any(source.dimension != self.cell.dimension for source in sources):
            raise ValueError("ghost source cells must match the target dimension")

        if source_kind is GhostSourceKind.PHYSICAL_BOUNDARY:
            if sources:
                raise ValueError("a physical-boundary ghost has no source cells")
            return
        if not sources:
            raise ValueError("an interior ghost needs at least one source cell")
        if source_kind is GhostSourceKind.SAME_LEVEL and (
            len(sources) != 1 or sources[0].level != self.cell.level
        ):
            raise ValueError("a same-level ghost needs one same-level source")
        if source_kind is GhostSourceKind.COARSE_PROLONGATION and (
            len(sources) != 1 or sources[0].level >= self.cell.level
        ):
            raise ValueError("coarse prolongation needs one coarser source")
        if source_kind is GhostSourceKind.FINE_RESTRICTION and any(
            source.level <= self.cell.level for source in sources
        ):
            raise ValueError("fine restriction needs only finer source cells")

    @property
    def boundary_label(self) -> str | None:
        """Return ``x-``, ``x+``, and so on for a physical boundary slot."""

        if self.source_kind is not GhostSourceKind.PHYSICAL_BOUNDARY:
            return None
        axis_name = ("x", "y", "z")[self.axis]
        return f"{axis_name}{'-' if self.side < 0 else '+'}"


class GhostTopology:
    """One ghost-source recipe for every side of every active leaf."""

    __slots__ = ("_lookup", "_lookup_view", "face_topology", "slots")

    def __init__(self, face_topology: FaceTopology) -> None:
        if not isinstance(face_topology, FaceTopology):
            raise TypeError("ghost topology requires a FaceTopology")
        slots: list[GhostSlot] = []
        for cell in face_topology.layout.cells:
            for axis in range(face_topology.layout.dimension):
                for side in (-1, 1):
                    faces = face_topology.cell_side_faces(cell, axis, side)
                    neighbors = tuple(
                        sorted(
                            {
                                neighbor
                                for face in faces
                                if (neighbor := face.other_cell(cell)) is not None
                            }
                        )
                    )
                    if not neighbors:
                        source_kind = GhostSourceKind.PHYSICAL_BOUNDARY
                    elif all(neighbor.level == cell.level for neighbor in neighbors):
                        source_kind = GhostSourceKind.SAME_LEVEL
                    elif all(neighbor.level < cell.level for neighbor in neighbors):
                        source_kind = GhostSourceKind.COARSE_PROLONGATION
                    elif all(neighbor.level > cell.level for neighbor in neighbors):
                        source_kind = GhostSourceKind.FINE_RESTRICTION
                    else:
                        raise RuntimeError(
                            "one cell side has an unsupported mixture of "
                            "neighbor levels"
                        )
                    slots.append(
                        GhostSlot(
                            cell=cell,
                            axis=axis,
                            side=side,
                            source_kind=source_kind,
                            source_cells=neighbors,
                            face_keys=tuple(sorted(face.key for face in faces)),
                        )
                    )

        lookup = {(slot.cell, slot.axis, slot.side): slot for slot in slots}
        if len(lookup) != len(slots):
            raise RuntimeError("ghost slot keys must be unique")
        self.face_topology = face_topology
        self.slots = tuple(slots)
        self._lookup = lookup
        self._lookup_view = MappingProxyType(lookup)

    @property
    def topology_id(self) -> str:
        return self.face_topology.topology_id

    @property
    def lookup(self) -> Mapping[tuple[Cell, int, int], GhostSlot]:
        return self._lookup_view

    def at(self, cell: Cell, axis: int, side: int) -> GhostSlot:
        try:
            return self._lookup[(cell, axis, side)]
        except KeyError as error:
            raise KeyError(
                "no ghost slot exists for the requested cell side"
            ) from error

    def by_source(self, source_kind: GhostSourceKind) -> tuple[GhostSlot, ...]:
        source_kind = GhostSourceKind(source_kind)
        return tuple(slot for slot in self.slots if slot.source_kind is source_kind)
