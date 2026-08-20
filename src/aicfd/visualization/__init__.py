"""Dependency-light visualization backends for inspecting discretizations."""

from aicfd.visualization.svg import (
    CellColorMode,
    SvgOptions,
    render_geometry_svg,
    write_geometry_svg,
)

__all__ = [
    "CellColorMode",
    "SvgOptions",
    "render_geometry_svg",
    "write_geometry_svg",
]
