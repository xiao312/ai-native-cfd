"""Small reference finite-volume operators."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from aicfd.fields import FaceField
from aicfd.representation import Cell


def _positive_cell_measures(
    field: FaceField,
    measures: Mapping[Cell, float] | None,
) -> NDArray[np.float64]:
    layout = field.topology.layout
    if measures is None:
        values = np.array(
            [layout.cell_measure(cell) for cell in layout.cells],
            dtype=np.float64,
        )
    else:
        if set(measures) != set(layout.cells):
            raise ValueError("cell measures must contain exactly the active leaves")
        values = np.array(
            [measures[cell] for cell in layout.cells],
            dtype=np.float64,
        )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("finite-volume cell measures must be finite and positive")
    return values


def flux_divergence(
    flux: FaceField,
    *,
    cell_measures: Mapping[Cell, float] | None = None,
) -> NDArray[np.float64]:
    """Return cell divergence from oriented normal face fluxes.

    Boundary-face values are outward fluxes because boundary normals point out of
    the domain. Interior values are positive from owner to neighbor. The result
    has units of integrated flux rate divided by cell measure.
    """

    rates = np.asarray(flux.integrated_rates(), dtype=np.float64)
    scalar = rates.ndim == 1
    if scalar:
        rates = rates[:, np.newaxis]
    divergence = np.zeros(
        (len(flux.topology.layout), flux.spec.n_components),
        dtype=np.float64,
    )
    layout = flux.topology.layout
    for face_index, face in enumerate(flux.topology.faces):
        divergence[layout.index(face.owner)] += rates[face_index]
        if face.neighbor is not None:
            divergence[layout.index(face.neighbor)] -= rates[face_index]

    measures = _positive_cell_measures(flux, cell_measures)
    divergence /= measures[:, np.newaxis]
    return divergence[:, 0] if scalar else divergence
