"""A dimension-independent identifier for an adaptive Cartesian cell."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product


@dataclass(frozen=True, order=True, slots=True)
class Cell:
    """Identify one cell in a binary, quad-, or octree.

    Parameters
    ----------
    level:
        Number of refinements between the root and this cell. The root is level 0.
    index:
        Integer coordinates at this level. In two dimensions, for example, a
        level-2 index is in ``{0, 1, 2, 3} x {0, 1, 2, 3}``.

    Notes
    -----
    Geometry is normalized to the unit box here. ``AdaptiveTree`` maps the unit
    box to a physical origin and extent.
    """

    level: int
    index: tuple[int, ...]

    def __post_init__(self) -> None:
        normalized_index = tuple(self.index)
        object.__setattr__(self, "index", normalized_index)

        if not isinstance(self.level, int) or isinstance(self.level, bool):
            raise TypeError("level must be an integer")
        if self.level < 0:
            raise ValueError("level must be non-negative")
        if not 1 <= len(normalized_index) <= 3:
            raise ValueError("cell dimension must be 1, 2, or 3")

        cells_per_axis = 1 << self.level
        for coordinate in normalized_index:
            if not isinstance(coordinate, int) or isinstance(coordinate, bool):
                raise TypeError("cell indices must be integers")
            if not 0 <= coordinate < cells_per_axis:
                raise ValueError(
                    f"index {normalized_index} is outside level {self.level}"
                )

    @property
    def dimension(self) -> int:
        """Number of spatial dimensions."""

        return len(self.index)

    @property
    def parent(self) -> Cell | None:
        """Return the parent cell, or ``None`` for the root."""

        if self.level == 0:
            return None
        return Cell(self.level - 1, tuple(value // 2 for value in self.index))

    def children(self) -> tuple[Cell, ...]:
        """Return the ``2**dimension`` children in deterministic order."""

        child_level = self.level + 1
        return tuple(
            Cell(
                child_level,
                tuple(
                    2 * value + offset
                    for value, offset in zip(self.index, bits, strict=True)
                ),
            )
            for bits in product((0, 1), repeat=self.dimension)
        )

    @property
    def morton_code(self) -> int:
        """Return the Morton path code within this cell's refinement level.

        At each level, one bit from every coordinate identifies which child was
        selected. Appending those child numbers produces a compact hierarchy-aware
        integer. The refinement level is still required for a globally unique ID.
        """

        code = 0
        for bit_position in range(self.level - 1, -1, -1):
            child_number = 0
            for axis, coordinate in enumerate(self.index):
                child_number |= ((coordinate >> bit_position) & 1) << axis
            code = (code << self.dimension) | child_number
        return code

    @property
    def stable_id(self) -> str:
        """Human-readable ID that is independent of array ordering."""

        return f"{self.dimension}D:L{self.level}:M{self.morton_code}"

    @property
    def normalized_bounds(self) -> tuple[tuple[float, float], ...]:
        """Cell bounds in the normalized unit interval along every axis."""

        denominator = 1 << self.level
        return tuple(
            (coordinate / denominator, (coordinate + 1) / denominator)
            for coordinate in self.index
        )

    @property
    def normalized_measure(self) -> Fraction:
        """Exact length, area, or volume inside the normalized unit box."""

        return Fraction(1, 1 << (self.dimension * self.level))

    def is_ancestor_of(self, other: Cell) -> bool:
        """Return whether this cell strictly contains ``other`` in the tree."""

        if self.dimension != other.dimension or self.level >= other.level:
            return False
        shift = other.level - self.level
        return self.index == tuple(value >> shift for value in other.index)
