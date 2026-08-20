"""Adapters for external simulation data."""

from aicfd.io.openfoam import (
    FoamField,
    OpenFoamImportReport,
    discover_time_directories,
    import_openfoam_cartesian_2d,
    read_foam_field,
)

__all__ = [
    "FoamField",
    "OpenFoamImportReport",
    "discover_time_directories",
    "import_openfoam_cartesian_2d",
    "read_foam_field",
]
