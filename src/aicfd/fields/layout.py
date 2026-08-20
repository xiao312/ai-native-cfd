"""Immutable array layout for one adaptive-tree leaf snapshot."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite, prod
from types import MappingProxyType
from typing import TYPE_CHECKING

from aicfd.representation import AdaptiveTree, Cell

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class TreeLayout:
    """Bind deterministic array rows to the leaves of one tree snapshot.

    A ``Cell`` remains the stable spatial identity. Its row number is only valid
    for this layout and may change after refinement or coarsening.
    """

    dimension: int
    origin: tuple[float, ...]
    extent: tuple[float, ...]
    cells: tuple[Cell, ...]
    topology_id: str = field(init=False)
    _cell_to_index: Mapping[Cell, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        origin = tuple(float(value) for value in self.origin)
        extent = tuple(float(value) for value in self.extent)
        cells = tuple(self.cells)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "extent", extent)
        object.__setattr__(self, "cells", cells)

        if self.dimension not in (1, 2, 3):
            raise ValueError("dimension must be 1, 2, or 3")
        if len(origin) != self.dimension or len(extent) != self.dimension:
            raise ValueError("origin and extent must match the layout dimension")
        if not all(isfinite(value) for value in (*origin, *extent)):
            raise ValueError("origin and extent must be finite")
        if any(length <= 0.0 for length in extent):
            raise ValueError("every extent must be positive")
        if not cells:
            raise ValueError("a layout needs at least one leaf cell")
        if any(cell.dimension != self.dimension for cell in cells):
            raise ValueError("every cell must match the layout dimension")
        if len(set(cells)) != len(cells):
            raise ValueError("layout cells must be unique")

        # Reconstructing a tree is inexpensive at this educational scale and
        # validates coverage and ancestor/leaf consistency in one place.
        tree = AdaptiveTree.from_dict(
            {
                "dimension": self.dimension,
                "origin": origin,
                "extent": extent,
                "leaves": [
                    {"level": cell.level, "index": cell.index} for cell in cells
                ],
            }
        )
        if tree.leaves != cells:
            raise ValueError("layout cells must use deterministic tree leaf order")

        lookup = MappingProxyType({cell: index for index, cell in enumerate(cells)})
        object.__setattr__(self, "_cell_to_index", lookup)
        object.__setattr__(self, "topology_id", self._fingerprint())

    @classmethod
    def from_tree(cls, tree: AdaptiveTree) -> TreeLayout:
        """Capture the current leaves and physical domain of ``tree``."""

        tree.assert_valid()
        return cls(
            dimension=tree.dimension,
            origin=tree.origin,
            extent=tree.extent,
            cells=tree.leaves,
        )

    @property
    def cell_to_index(self) -> Mapping[Cell, int]:
        """Read-only map from stable cell identity to array row."""

        return self._cell_to_index

    def __len__(self) -> int:
        return len(self.cells)

    def index(self, cell: Cell) -> int:
        """Return the row assigned to ``cell`` in this snapshot."""

        try:
            return self._cell_to_index[cell]
        except KeyError as error:
            raise KeyError(f"{cell} is not a leaf in this layout") from error

    def physical_bounds(self, cell: Cell) -> tuple[tuple[float, float], ...]:
        """Return physical bounds without reconstructing the mutable tree."""

        self.index(cell)
        return tuple(
            (start + length * lower, start + length * upper)
            for start, length, (lower, upper) in zip(
                self.origin, self.extent, cell.normalized_bounds, strict=True
            )
        )

    def cell_center(self, cell: Cell) -> tuple[float, ...]:
        """Return the physical centre of one layout cell."""

        return tuple(
            0.5 * (lower + upper) for lower, upper in self.physical_bounds(cell)
        )

    def cell_measure(self, cell: Cell) -> float:
        """Return the full Cartesian length, area, or volume of a cell."""

        self.index(cell)
        scale = 1 << cell.level
        return prod(length / scale for length in self.extent)

    def to_tree(self) -> AdaptiveTree:
        """Return a new mutable tree with this exact topology."""

        return AdaptiveTree.from_dict(
            {
                "dimension": self.dimension,
                "origin": self.origin,
                "extent": self.extent,
                "leaves": [
                    {"level": cell.level, "index": cell.index} for cell in self.cells
                ],
            }
        )

    def matches(self, tree: AdaptiveTree) -> bool:
        """Return whether ``tree`` has the same domain and ordered leaves."""

        return self.topology_id == TreeLayout.from_tree(tree).topology_id

    def _fingerprint(self) -> str:
        payload: dict[str, object] = {
            "dimension": self.dimension,
            "origin": [value.hex() for value in self.origin],
            "extent": [value.hex() for value in self.extent],
            "cells": [[cell.level, *cell.index] for cell in self.cells],
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        return sha256(encoded).hexdigest()
