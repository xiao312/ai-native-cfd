"""Dense values bound to explicit adaptive face-segment rows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aicfd.fields.field import FieldLocation, FieldSpec, ValueRepresentation
from aicfd.representation.face import FaceKey, FaceSegment, FaceTopology


@dataclass(slots=True)
class FaceField:
    """A scalar or multi-component value on every oriented face segment.

    For a normal flux, positive values follow ``FaceSegment.normal``. A
    ``FACE_AVERAGE`` value is multiplied by segment area before it enters a
    finite-volume balance; a ``FACE_INTEGRAL`` value already includes area.
    """

    spec: FieldSpec
    topology: FaceTopology
    values: NDArray[np.floating]

    def __init__(
        self,
        spec: FieldSpec,
        topology: FaceTopology,
        values: ArrayLike,
    ) -> None:
        if spec.location is not FieldLocation.FACE:
            raise ValueError("FaceField requires FieldLocation.FACE")
        if spec.value_representation not in (
            ValueRepresentation.FACE_AVERAGE,
            ValueRepresentation.FACE_INTEGRAL,
        ):
            raise ValueError("FaceField requires a face-compatible representation")

        array = np.array(values, dtype=spec.dtype, copy=True)
        expected_shape = (
            (len(topology),)
            if spec.n_components == 1
            else (len(topology), spec.n_components)
        )
        if array.shape != expected_shape:
            raise ValueError(
                f"face field {spec.name!r} needs shape {expected_shape}, "
                f"got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"face field {spec.name!r} contains non-finite values")
        if spec.lower_bound is not None and np.any(array < spec.lower_bound):
            raise ValueError(f"face field {spec.name!r} is below its lower bound")
        if spec.upper_bound is not None and np.any(array > spec.upper_bound):
            raise ValueError(f"face field {spec.name!r} is above its upper bound")

        self.spec = spec
        self.topology = topology
        self.values = array

    @property
    def topology_id(self) -> str:
        return self.topology.topology_id

    def at(
        self,
        face: FaceKey | FaceSegment,
    ) -> float | NDArray[np.floating]:
        value = self.values[self.topology.index(face)]
        if self.spec.n_components == 1:
            return float(value)
        return value.copy()

    def copy(self) -> FaceField:
        return FaceField(self.spec, self.topology, self.values)

    def component(self, component: int | str = 0) -> NDArray[np.floating]:
        if isinstance(component, str):
            try:
                component = self.spec.component_names.index(component)
            except ValueError as error:
                raise KeyError(
                    f"face field {self.spec.name!r} has no {component!r} component"
                ) from error
        if not isinstance(component, int) or isinstance(component, bool):
            raise TypeError("component must be an integer index or component name")
        if not 0 <= component < self.spec.n_components:
            raise IndexError("face-field component is out of range")
        if self.spec.n_components == 1:
            return self.values
        return self.values[:, component]

    def integrated_rates(self) -> NDArray[np.floating]:
        """Return values with face area included, preserving component shape."""

        rates = np.array(self.values, copy=True)
        if self.spec.value_representation is ValueRepresentation.FACE_AVERAGE:
            areas = np.array(
                [face.area for face in self.topology.faces],
                dtype=self.spec.dtype,
            )
            if rates.ndim == 1:
                rates *= areas
            else:
                rates *= areas[:, np.newaxis]
        return rates
