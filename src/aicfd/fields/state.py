"""A collection of fields on one immutable tree layout."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import isfinite
from types import MappingProxyType

from aicfd.fields.field import CellField
from aicfd.fields.layout import TreeLayout
from aicfd.representation import AdaptiveTree


class State:
    """Physical fields, time, and step attached to one topology snapshot."""

    __slots__ = ("_fields", "_fields_view", "layout", "step", "time")

    def __init__(
        self,
        layout: TreeLayout,
        fields: Iterable[CellField] | Mapping[str, CellField] = (),
        *,
        time: float = 0.0,
        step: int = 0,
    ) -> None:
        time = float(time)
        if not isfinite(time):
            raise ValueError("state time must be finite")
        if not isinstance(step, int) or isinstance(step, bool):
            raise TypeError("state step must be an integer")
        if step < 0:
            raise ValueError("state step must be non-negative")

        if isinstance(fields, Mapping):
            candidates = tuple(fields.items())
        else:
            candidates = tuple((field.spec.name, field) for field in fields)

        registry: dict[str, CellField] = {}
        for name, field in candidates:
            if not isinstance(field, CellField):
                raise TypeError("state fields must be CellField instances")
            if name != field.spec.name:
                raise ValueError("field registry key must match FieldSpec.name")
            if name in registry:
                raise ValueError(f"duplicate field name {name!r}")
            if field.layout_id != layout.topology_id:
                raise ValueError(
                    f"field {name!r} belongs to a stale or different tree layout"
                )
            registry[name] = field

        self.layout = layout
        self.time = time
        self.step = step
        self._fields = registry
        self._fields_view = MappingProxyType(registry)

    @property
    def fields(self) -> Mapping[str, CellField]:
        """Read-only registry keyed by field name."""

        return self._fields_view

    def __contains__(self, field_name: object) -> bool:
        return field_name in self._fields

    def __getitem__(self, field_name: str) -> CellField:
        try:
            return self._fields[field_name]
        except KeyError as error:
            raise KeyError(f"state has no field named {field_name!r}") from error

    def to_tree(self) -> AdaptiveTree:
        """Return a mutable copy of the state's tree snapshot."""

        return self.layout.to_tree()

    def with_field(self, field: CellField, *, replace: bool = False) -> State:
        """Return a new registry containing ``field`` without mutating this one."""

        if field.layout_id != self.layout.topology_id:
            raise ValueError("new field belongs to a stale or different tree layout")
        if field.spec.name in self._fields and not replace:
            raise ValueError(f"field {field.spec.name!r} already exists")
        fields = dict(self._fields)
        fields[field.spec.name] = field
        return State(self.layout, fields, time=self.time, step=self.step)
