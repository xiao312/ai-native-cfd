"""Immutable level views over one adaptive-tree leaf snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from aicfd.representation.cell import Cell

if TYPE_CHECKING:
    from collections.abc import Mapping

    from aicfd.fields.layout import TreeLayout


@dataclass(frozen=True, slots=True)
class AMRLevel:
    """Structural cells present at one refinement level.

    ``active_cells`` are leaves with physical unknowns. ``refined_cells`` are
    covered parents retained only to describe the hierarchy.
    """

    number: int
    cells: tuple[Cell, ...]
    active_cells: tuple[Cell, ...]
    refined_cells: tuple[Cell, ...]

    @property
    def is_empty(self) -> bool:
        return not self.cells


class AMRHierarchy:
    """Parent/child and per-level structure derived from a :class:`TreeLayout`.

    Physical fields remain attached only to the active leaves in ``layout``.
    Covered coarse cells are structural nodes; duplicating state on them would
    require an explicit multilevel time-integration policy.
    """

    __slots__ = (
        "_children",
        "_children_view",
        "_node_set",
        "_parent",
        "_parent_view",
        "layout",
        "levels",
        "nodes",
    )

    def __init__(self, layout: TreeLayout) -> None:
        tree = layout.to_tree()
        nodes = tree.nodes
        node_set = frozenset(nodes)
        active = frozenset(layout.cells)
        levels: list[AMRLevel] = []
        for number in range(tree.max_level + 1):
            level_cells = tuple(cell for cell in nodes if cell.level == number)
            levels.append(
                AMRLevel(
                    number=number,
                    cells=level_cells,
                    active_cells=tuple(cell for cell in level_cells if cell in active),
                    refined_cells=tuple(
                        cell for cell in level_cells if cell not in active
                    ),
                )
            )

        children: dict[Cell, tuple[Cell, ...]] = {}
        parent: dict[Cell, Cell | None] = {}
        for node in nodes:
            parent[node] = node.parent
            children[node] = tuple(
                child for child in node.children() if child in node_set
            )
            if node in active and children[node]:
                raise RuntimeError("an active leaf cannot have materialized children")
            if node not in active and len(children[node]) != 2**layout.dimension:
                raise RuntimeError("a refined hierarchy node needs every child")

        self.layout = layout
        self.nodes = nodes
        self.levels = tuple(levels)
        self._node_set = node_set
        self._children = children
        self._children_view = MappingProxyType(children)
        self._parent = parent
        self._parent_view = MappingProxyType(parent)

    @property
    def topology_id(self) -> str:
        """Fingerprint of the active leaf topology represented here."""

        return self.layout.topology_id

    @property
    def max_level(self) -> int:
        return len(self.levels) - 1

    @property
    def active_cells(self) -> tuple[Cell, ...]:
        return self.layout.cells

    @property
    def refined_cells(self) -> tuple[Cell, ...]:
        return tuple(
            cell for cell in self.nodes if cell not in self.layout.cell_to_index
        )

    @property
    def hierarchy_edges(self) -> tuple[tuple[Cell, Cell], ...]:
        return tuple(
            (parent, child)
            for parent, children in self._children.items()
            for child in children
        )

    @property
    def children(self) -> Mapping[Cell, tuple[Cell, ...]]:
        return self._children_view

    @property
    def parents(self) -> Mapping[Cell, Cell | None]:
        return self._parent_view

    def at_level(self, number: int) -> AMRLevel:
        """Return one level, including empty structural categories."""

        if not isinstance(number, int) or isinstance(number, bool):
            raise TypeError("level number must be an integer")
        if not 0 <= number <= self.max_level:
            raise IndexError("hierarchy level is out of range")
        return self.levels[number]

    def children_of(self, cell: Cell) -> tuple[Cell, ...]:
        try:
            return self._children[cell]
        except KeyError as error:
            raise KeyError(f"{cell} is not materialized in this hierarchy") from error

    def parent_of(self, cell: Cell) -> Cell | None:
        try:
            return self._parent[cell]
        except KeyError as error:
            raise KeyError(f"{cell} is not materialized in this hierarchy") from error

    def is_active(self, cell: Cell) -> bool:
        if cell not in self._node_set:
            raise KeyError(f"{cell} is not materialized in this hierarchy")
        return cell in self.layout.cell_to_index
