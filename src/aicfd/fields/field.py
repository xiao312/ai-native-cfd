"""Metadata and NumPy storage for physical fields."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aicfd.fields.layout import TreeLayout
from aicfd.representation import Cell


class FieldLocation(str, Enum):
    """Where a field value is located relative to a mesh element."""

    CELL = "cell"
    FACE = "face"
    FACE_X = "face_x"
    FACE_Y = "face_y"
    FACE_Z = "face_z"
    NODE = "node"
    EMBEDDED_BOUNDARY = "embedded_boundary"


class ValueRepresentation(str, Enum):
    """Numerical meaning used when a field changes resolution."""

    CELL_AVERAGE = "cell_average"
    CELL_INTEGRAL = "cell_integral"
    FACE_AVERAGE = "face_average"
    FACE_INTEGRAL = "face_integral"
    POINT_VALUE = "point_value"


_VALID_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Describe the components, placement, units, and transfer meaning."""

    name: str
    component_names: tuple[str, ...] = ("value",)
    location: FieldLocation = FieldLocation.CELL
    value_representation: ValueRepresentation = ValueRepresentation.CELL_AVERAGE
    dtype: str = "float64"
    unit: str = "1"
    unit_dimensions: tuple[float, ...] | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        if not _VALID_NAME.fullmatch(self.name):
            raise ValueError(
                "field name must start with a letter or underscore and contain "
                "only letters, numbers, '_', '.', or '-'"
            )

        components = tuple(self.component_names)
        if not components or any(
            not _VALID_NAME.fullmatch(name) for name in components
        ):
            raise ValueError("component_names must contain valid, non-empty names")
        if len(set(components)) != len(components):
            raise ValueError("component names must be unique")
        object.__setattr__(self, "component_names", components)

        try:
            location = FieldLocation(self.location)
            representation = ValueRepresentation(self.value_representation)
        except ValueError as error:
            raise ValueError(
                "unknown field location or value representation"
            ) from error
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "value_representation", representation)

        dtype = np.dtype(self.dtype)
        if dtype.kind != "f":
            raise ValueError("physical field dtype must be floating point")
        object.__setattr__(self, "dtype", dtype.name)

        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError("unit must be a non-empty string")
        if self.unit_dimensions is not None:
            dimensions = tuple(float(value) for value in self.unit_dimensions)
            if len(dimensions) != 7 or not all(isfinite(value) for value in dimensions):
                raise ValueError("unit_dimensions must contain seven finite SI powers")
            object.__setattr__(self, "unit_dimensions", dimensions)

        for name, bound in (
            ("lower_bound", self.lower_bound),
            ("upper_bound", self.upper_bound),
        ):
            if bound is not None and not isfinite(bound):
                raise ValueError(f"{name} must be finite when provided")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("lower_bound must not exceed upper_bound")

    @property
    def n_components(self) -> int:
        """Number of scalar components stored for each location."""

        return len(self.component_names)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to JSON-compatible containers."""

        return {
            "name": self.name,
            "component_names": list(self.component_names),
            "location": self.location.value,
            "value_representation": self.value_representation.value,
            "dtype": self.dtype,
            "unit": self.unit,
            "unit_dimensions": (
                None if self.unit_dimensions is None else list(self.unit_dimensions)
            ),
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldSpec:
        """Restore metadata produced by :meth:`to_dict`."""

        return cls(
            name=data["name"],
            component_names=tuple(data["component_names"]),
            location=FieldLocation(data["location"]),
            value_representation=ValueRepresentation(data["value_representation"]),
            dtype=data["dtype"],
            unit=data["unit"],
            unit_dimensions=(
                None
                if data.get("unit_dimensions") is None
                else tuple(data["unit_dimensions"])
            ),
            lower_bound=data.get("lower_bound"),
            upper_bound=data.get("upper_bound"),
        )


@dataclass(slots=True)
class CellField:
    """Dense cell data whose rows are bound to one :class:`TreeLayout`."""

    spec: FieldSpec
    layout: TreeLayout
    values: NDArray[np.floating]

    def __init__(
        self,
        spec: FieldSpec,
        layout: TreeLayout,
        values: ArrayLike,
    ) -> None:
        if spec.location is not FieldLocation.CELL:
            raise ValueError("CellField currently supports only cell-located data")
        if spec.value_representation not in (
            ValueRepresentation.CELL_AVERAGE,
            ValueRepresentation.CELL_INTEGRAL,
            ValueRepresentation.POINT_VALUE,
        ):
            raise ValueError("CellField requires a cell-compatible representation")

        array = np.array(values, dtype=spec.dtype, copy=True)
        expected_shape = (
            (len(layout),)
            if spec.n_components == 1
            else (len(layout), spec.n_components)
        )
        if array.shape != expected_shape:
            raise ValueError(
                f"field {spec.name!r} needs shape {expected_shape}, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"field {spec.name!r} contains non-finite values")
        if spec.lower_bound is not None and np.any(array < spec.lower_bound):
            raise ValueError(f"field {spec.name!r} is below its lower bound")
        if spec.upper_bound is not None and np.any(array > spec.upper_bound):
            raise ValueError(f"field {spec.name!r} is above its upper bound")

        self.spec = spec
        self.layout = layout
        self.values = array

    @property
    def layout_id(self) -> str:
        """Fingerprint of the topology to which array rows belong."""

        return self.layout.topology_id

    def at(self, cell: Cell) -> float | NDArray[np.floating]:
        """Return the value attached to ``cell`` in this layout."""

        value = self.values[self.layout.index(cell)]
        if self.spec.n_components == 1:
            return float(value)
        return value.copy()

    def copy(self) -> CellField:
        """Return an independent field with the same metadata and layout."""

        return CellField(self.spec, self.layout, self.values)

    def component(self, component: int | str = 0) -> NDArray[np.floating]:
        """Return a one-dimensional view of one named or numbered component."""

        if isinstance(component, str):
            try:
                component = self.spec.component_names.index(component)
            except ValueError as error:
                raise KeyError(
                    f"field {self.spec.name!r} has no {component!r} component"
                ) from error
        if not isinstance(component, int) or isinstance(component, bool):
            raise TypeError("component must be an integer index or component name")
        if not 0 <= component < self.spec.n_components:
            raise IndexError("field component is out of range")
        if self.spec.n_components == 1:
            return self.values
        return self.values[:, component]
