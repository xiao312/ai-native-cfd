"""Time-integrated coarse/fine flux mismatch and conservative refluxing."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from aicfd.fields import CellField, FaceField, ValueRepresentation
from aicfd.representation import Cell, FaceSegment, FaceTopology


class FluxRegister:
    """Accumulate coarse and fine flux estimates on coarse-fine face segments.

    Stored entries are time-and-area-integrated amounts. Fine levels may call
    :meth:`accumulate_fine` several times during subcycling; a coarse level usually
    calls :meth:`accumulate_coarse` once.
    """

    __slots__ = (
        "_coarse_integrals",
        "_face_to_row",
        "_fine_integrals",
        "interface_faces",
        "n_components",
        "topology",
    )

    def __init__(self, topology: FaceTopology, *, n_components: int = 1) -> None:
        if not isinstance(topology, FaceTopology):
            raise TypeError("flux register requires a FaceTopology")
        if not isinstance(n_components, int) or isinstance(n_components, bool):
            raise TypeError("flux-register component count must be an integer")
        if n_components <= 0:
            raise ValueError("flux-register component count must be positive")

        interface_faces = topology.coarse_fine_faces
        self.topology = topology
        self.n_components = n_components
        self.interface_faces = interface_faces
        self._face_to_row = {face.key: row for row, face in enumerate(interface_faces)}
        shape = (len(interface_faces), n_components)
        self._coarse_integrals = np.zeros(shape, dtype=np.float64)
        self._fine_integrals = np.zeros(shape, dtype=np.float64)

    @property
    def coarse_integrals(self) -> NDArray[np.float64]:
        return self._coarse_integrals.copy()

    @property
    def fine_integrals(self) -> NDArray[np.float64]:
        return self._fine_integrals.copy()

    @property
    def mismatch(self) -> NDArray[np.float64]:
        """Fine-minus-coarse time-integrated flux in face orientation."""

        return self._fine_integrals - self._coarse_integrals

    @property
    def is_empty(self) -> bool:
        return not np.any(self._coarse_integrals) and not np.any(self._fine_integrals)

    def clear(self) -> None:
        self._coarse_integrals.fill(0.0)
        self._fine_integrals.fill(0.0)

    @staticmethod
    def _time_step(delta_t: float) -> float:
        delta_t = float(delta_t)
        if not np.isfinite(delta_t) or delta_t <= 0.0:
            raise ValueError("flux accumulation time step must be finite and positive")
        return delta_t

    def _accumulate(
        self,
        target: NDArray[np.float64],
        flux: FaceField,
        delta_t: float,
    ) -> None:
        if flux.topology_id != self.topology.topology_id:
            raise ValueError("face flux belongs to a stale or different topology")
        if flux.spec.n_components != self.n_components:
            raise ValueError("face flux and register component counts differ")
        delta_t = self._time_step(delta_t)
        rates = np.asarray(flux.integrated_rates(), dtype=np.float64)
        if rates.ndim == 1:
            rates = rates[:, np.newaxis]
        rows = np.array(
            [self.topology.index(face) for face in self.interface_faces],
            dtype=np.int64,
        )
        target += delta_t * rates[rows]

    def accumulate_coarse(self, flux: FaceField, delta_t: float) -> None:
        """Add one coarse-level flux contribution."""

        self._accumulate(self._coarse_integrals, flux, delta_t)

    def accumulate_fine(self, flux: FaceField, delta_t: float) -> None:
        """Add one fine-level flux contribution."""

        self._accumulate(self._fine_integrals, flux, delta_t)

    def row(self, face: FaceSegment) -> int:
        try:
            return self._face_to_row[face.key]
        except KeyError as error:
            raise KeyError("flux registers contain only coarse-fine faces") from error

    def reflux(
        self,
        field: CellField,
        *,
        cell_measures: Mapping[Cell, float] | None = None,
        clear_after: bool = False,
    ) -> CellField:
        """Return ``field`` with conservative mismatch corrections on coarse cells.

        Fine cells are assumed to have already used the fine flux. Refluxing
        changes only the adjacent coarse cells so the composite active-leaf total
        matches the fine interface flux.
        """

        if field.layout_id != self.topology.layout.topology_id:
            raise ValueError("cell field and flux register use different topologies")
        if field.spec.n_components != self.n_components:
            raise ValueError("cell field and register component counts differ")
        if field.spec.value_representation is ValueRepresentation.POINT_VALUE:
            raise ValueError("point values cannot receive a conservative reflux")
        if field.spec.value_representation not in (
            ValueRepresentation.CELL_AVERAGE,
            ValueRepresentation.CELL_INTEGRAL,
        ):
            raise ValueError("refluxing requires cell averages or cell integrals")

        layout = field.layout
        if cell_measures is None:
            measures = {cell: layout.cell_measure(cell) for cell in layout.cells}
        else:
            if set(cell_measures) != set(layout.cells):
                raise ValueError("cell measures must contain exactly the active leaves")
            measures = {cell: float(cell_measures[cell]) for cell in layout.cells}
        if any(
            not np.isfinite(measure) or measure <= 0.0 for measure in measures.values()
        ):
            raise ValueError("reflux cell measures must be finite and positive")

        values = np.array(field.values, copy=True)
        scalar = values.ndim == 1
        if scalar:
            values = values[:, np.newaxis]
        mismatch = self.mismatch
        for register_row, face in enumerate(self.interface_faces):
            coarse_cell = face.coarse_cell
            assert coarse_cell is not None
            extensive_correction = (
                -face.incidence_sign(coarse_cell) * mismatch[register_row]
            )
            if field.spec.value_representation is ValueRepresentation.CELL_AVERAGE:
                extensive_correction /= measures[coarse_cell]
            values[layout.index(coarse_cell)] += extensive_correction

        corrected = CellField(
            field.spec,
            layout,
            values[:, 0] if scalar else values,
        )
        if clear_after:
            self.clear()
        return corrected
