"""A small, readable adaptive Cartesian tree implementation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from math import prod

from aicfd.representation.cell import Cell


class AdaptiveTree:
    """Represent an adaptive Cartesian mesh by its leaf cells.

    ``dimension=1`` creates a binary tree, ``dimension=2`` a quadtree, and
    ``dimension=3`` an octree. Refinement is isotropic: every axis is halved.

    This first implementation is serial and intentionally favors clarity. Face
    neighbours are found by comparing every pair of leaves, which is O(N^2).
    """

    def __init__(
        self,
        dimension: int,
        origin: Sequence[float] | None = None,
        extent: Sequence[float] | None = None,
    ) -> None:
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise TypeError("dimension must be an integer")
        if dimension not in (1, 2, 3):
            raise ValueError("dimension must be 1, 2, or 3")

        self.dimension = dimension
        self.origin = self._coordinate_tuple(origin, default=0.0, name="origin")
        self.extent = self._coordinate_tuple(extent, default=1.0, name="extent")
        if any(length <= 0.0 for length in self.extent):
            raise ValueError("every extent must be positive")

        self._root = Cell(0, (0,) * dimension)
        self._leaves: set[Cell] = {self._root}

    def _coordinate_tuple(
        self,
        values: Sequence[float] | None,
        *,
        default: float,
        name: str,
    ) -> tuple[float, ...]:
        result = (default,) * self.dimension if values is None else tuple(values)
        if len(result) != self.dimension:
            raise ValueError(f"{name} must have {self.dimension} entries")
        try:
            return tuple(float(value) for value in result)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} entries must be real numbers") from error

    @property
    def root(self) -> Cell:
        """The single level-0 cell covering the complete domain."""

        return self._root

    @property
    def max_level(self) -> int:
        """Deepest refinement level currently used by a leaf."""

        return max(cell.level for cell in self._leaves)

    @property
    def leaves(self) -> tuple[Cell, ...]:
        """Current computational cells in deterministic Morton-prefix order."""

        max_level = self.max_level
        return tuple(
            sorted(
                self._leaves,
                key=lambda cell: (
                    cell.morton_code << (self.dimension * (max_level - cell.level)),
                    cell.level,
                    cell.index,
                ),
            )
        )

    @property
    def nodes(self) -> tuple[Cell, ...]:
        """All leaf and inferred internal cells in the hierarchy."""

        nodes: set[Cell] = set()
        for leaf in self._leaves:
            current: Cell | None = leaf
            while current is not None:
                nodes.add(current)
                current = current.parent
        return tuple(sorted(nodes))

    @property
    def hierarchy_edges(self) -> tuple[tuple[Cell, Cell], ...]:
        """Directed ``(parent, child)`` relationships for all materialized nodes."""

        return tuple(
            (parent, node) for node in self.nodes if (parent := node.parent) is not None
        )

    @property
    def normalized_leaf_measure(self) -> Fraction:
        """Exact total normalized measure covered by all leaves."""

        return sum(
            (cell.normalized_measure for cell in self._leaves), start=Fraction(0)
        )

    def __len__(self) -> int:
        """Number of leaf cells."""

        return len(self._leaves)

    def __contains__(self, cell: object) -> bool:
        return cell in self._leaves

    def __repr__(self) -> str:
        return (
            f"AdaptiveTree(dimension={self.dimension}, leaves={len(self)}, "
            f"max_level={self.max_level})"
        )

    def _require_compatible(self, cell: Cell) -> None:
        if not isinstance(cell, Cell):
            raise TypeError("expected a Cell")
        if cell.dimension != self.dimension:
            raise ValueError("cell dimension does not match the tree")

    def refine(self, cell: Cell) -> tuple[Cell, ...]:
        """Replace one leaf with all of its children."""

        self._require_compatible(cell)
        if cell not in self._leaves:
            raise ValueError(f"cannot refine non-leaf cell {cell}")

        children = cell.children()
        self._leaves.remove(cell)
        self._leaves.update(children)
        return children

    def refine_many(self, cells: Iterable[Cell]) -> tuple[Cell, ...]:
        """Refine several cells that are leaves at the start of the operation."""

        targets = tuple(dict.fromkeys(cells))
        for cell in targets:
            self._require_compatible(cell)
            if cell not in self._leaves:
                raise ValueError(f"cannot refine non-leaf cell {cell}")

        children: list[Cell] = []
        for cell in targets:
            children.extend(self.refine(cell))
        return tuple(children)

    def can_coarsen(self, parent: Cell) -> bool:
        """Return whether every child of ``parent`` is currently a leaf."""

        self._require_compatible(parent)
        return all(child in self._leaves for child in parent.children())

    def coarsen(self, parent: Cell) -> Cell:
        """Replace a complete family of leaf children with their parent."""

        self._require_compatible(parent)
        children = parent.children()
        if not all(child in self._leaves for child in children):
            raise ValueError("coarsening requires every child to be a leaf")

        self._leaves.difference_update(children)
        self._leaves.add(parent)
        return parent

    def physical_bounds(self, cell: Cell) -> tuple[tuple[float, float], ...]:
        """Return the physical lower and upper coordinate on every axis."""

        self._require_compatible(cell)
        return tuple(
            (
                start + length * lower,
                start + length * upper,
            )
            for start, length, (lower, upper) in zip(
                self.origin, self.extent, cell.normalized_bounds, strict=True
            )
        )

    def cell_center(self, cell: Cell) -> tuple[float, ...]:
        """Return the physical cell centre."""

        return tuple(
            0.5 * (lower + upper) for lower, upper in self.physical_bounds(cell)
        )

    def cell_size(self, cell: Cell) -> tuple[float, ...]:
        """Return the physical width along every axis."""

        self._require_compatible(cell)
        scale = 1 << cell.level
        return tuple(length / scale for length in self.extent)

    def cell_measure(self, cell: Cell) -> float:
        """Return physical length in 1D, area in 2D, or volume in 3D."""

        return prod(self.cell_size(cell))

    def locate(self, point: Sequence[float]) -> Cell:
        """Return the leaf containing a physical point.

        The upper domain boundary belongs to the final cell on that axis. This is
        convenient for user-facing queries even though numerical grids often use
        half-open intervals internally.
        """

        coordinates = tuple(float(value) for value in point)
        if len(coordinates) != self.dimension:
            raise ValueError(f"point must have {self.dimension} entries")

        normalized: list[float] = []
        for value, start, length in zip(
            coordinates, self.origin, self.extent, strict=True
        ):
            upper = start + length
            if not start <= value <= upper:
                raise ValueError("point lies outside the tree domain")
            normalized.append((value - start) / length)

        for level in range(self.max_level + 1):
            cells_per_axis = 1 << level
            index = tuple(
                min(int(value * cells_per_axis), cells_per_axis - 1)
                for value in normalized
            )
            candidate = Cell(level, index)
            if candidate in self._leaves:
                return candidate

        raise RuntimeError("tree leaves do not cover the domain")

    @staticmethod
    def _integer_bounds(cell: Cell, common_level: int) -> tuple[tuple[int, int], ...]:
        scale = 1 << (common_level - cell.level)
        return tuple(
            (coordinate * scale, (coordinate + 1) * scale) for coordinate in cell.index
        )

    @classmethod
    def _share_face(cls, first: Cell, second: Cell) -> bool:
        """Return whether two dyadic cells share a face with positive measure."""

        if first.dimension != second.dimension or first == second:
            return False

        common_level = max(first.level, second.level)
        first_bounds = cls._integer_bounds(first, common_level)
        second_bounds = cls._integer_bounds(second, common_level)
        touching_axes = 0

        for (first_lower, first_upper), (second_lower, second_upper) in zip(
            first_bounds, second_bounds, strict=True
        ):
            if first_upper == second_lower or second_upper == first_lower:
                touching_axes += 1
                continue
            overlap = min(first_upper, second_upper) - max(first_lower, second_lower)
            if overlap <= 0:
                return False

        return touching_axes == 1

    def face_neighbors(self, cell: Cell) -> tuple[Cell, ...]:
        """Return all leaves sharing a face with ``cell``.

        A coarse cell may have multiple finer neighbours along one face.
        """

        self._require_compatible(cell)
        if cell not in self._leaves:
            raise ValueError("face neighbours are defined for leaf cells")
        return tuple(
            candidate for candidate in self.leaves if self._share_face(cell, candidate)
        )

    def is_balanced(self) -> bool:
        """Return whether face neighbours differ by at most one level."""

        leaves = self.leaves
        return all(
            not self._share_face(first, second) or abs(first.level - second.level) <= 1
            for position, first in enumerate(leaves)
            for second in leaves[position + 1 :]
        )

    def balance(self) -> tuple[Cell, ...]:
        """Refine coarse leaves until the tree satisfies 2:1 face balance.

        The method never coarsens cells. It returns the coarse cells that had to be
        refined in addition to any user-requested refinement.
        """

        refined: list[Cell] = []
        while True:
            leaves = self.leaves
            to_refine: set[Cell] = set()
            for position, first in enumerate(leaves):
                for second in leaves[position + 1 :]:
                    if not self._share_face(first, second):
                        continue
                    if first.level + 1 < second.level:
                        to_refine.add(first)
                    elif second.level + 1 < first.level:
                        to_refine.add(second)

            if not to_refine:
                return tuple(refined)

            for cell in sorted(to_refine):
                if cell in self._leaves:
                    self.refine(cell)
                    refined.append(cell)

    def assert_valid(self) -> None:
        """Raise an error if the leaf set violates basic tree invariants."""

        if not self._leaves:
            raise RuntimeError("a tree must contain at least one leaf")

        for leaf in self._leaves:
            self._require_compatible(leaf)
            ancestor = leaf.parent
            while ancestor is not None:
                if ancestor in self._leaves:
                    raise RuntimeError("a leaf cannot contain another leaf")
                ancestor = ancestor.parent

        if self.normalized_leaf_measure != 1:
            raise RuntimeError("leaf cells do not cover the root exactly")

    def to_dict(self) -> dict[str, object]:
        """Serialize the topology to plain Python containers."""

        return {
            "dimension": self.dimension,
            "origin": list(self.origin),
            "extent": list(self.extent),
            "leaves": [
                {"level": cell.level, "index": list(cell.index)} for cell in self.leaves
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> AdaptiveTree:
        """Construct and validate a tree produced by :meth:`to_dict`."""

        dimension = int(data["dimension"])
        origin = data["origin"]
        extent = data["extent"]
        leaves = data["leaves"]
        if not isinstance(origin, Sequence) or isinstance(origin, (str, bytes)):
            raise TypeError("origin must be a sequence")
        if not isinstance(extent, Sequence) or isinstance(extent, (str, bytes)):
            raise TypeError("extent must be a sequence")
        if not isinstance(leaves, Sequence) or isinstance(leaves, (str, bytes)):
            raise TypeError("leaves must be a sequence")

        tree = cls(dimension=dimension, origin=origin, extent=extent)
        restored: set[Cell] = set()
        for item in leaves:
            if not isinstance(item, Mapping):
                raise TypeError("each leaf must be a mapping")
            index = item["index"]
            if not isinstance(index, Sequence) or isinstance(index, (str, bytes)):
                raise TypeError("leaf index must be a sequence")
            restored.add(Cell(level=int(item["level"]), index=tuple(index)))

        tree._leaves = restored
        tree.assert_valid()
        return tree
