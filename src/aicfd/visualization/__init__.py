"""Dependency-light visualization backends for inspecting discretizations."""

from aicfd.visualization.field_svg import (
    FieldSvgOptions,
    render_field_svg,
    write_field_svg,
)
from aicfd.visualization.svg import (
    CellColorMode,
    SvgOptions,
    render_geometry_svg,
    write_geometry_svg,
)

__all__ = [
    "CellColorMode",
    "FieldSvgOptions",
    "SvgOptions",
    "render_field_svg",
    "render_geometry_svg",
    "write_field_svg",
    "write_geometry_svg",
]
