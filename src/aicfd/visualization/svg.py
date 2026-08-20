"""Pure-Python SVG rendering for two-dimensional adaptive geometry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from html import escape
from pathlib import Path

from shapely.geometry import LinearRing, LineString
from shapely.geometry.base import BaseGeometry

from aicfd.geometry import (
    CellClassification,
    Obstacle2D,
    SnappedCell2D,
    SnappedGeometry2D,
)
from aicfd.representation import AdaptiveTree, Cell

Point2D = tuple[float, float]

_LEVEL_COLORS = (
    "#eff6ff",
    "#dbeafe",
    "#bfdbfe",
    "#93c5fd",
    "#60a5fa",
    "#3b82f6",
    "#2563eb",
    "#1d4ed8",
    "#1e40af",
    "#1e3a8a",
)
_CLASSIFICATION_COLORS = {
    CellClassification.FLUID: "#ffffff",
    CellClassification.CUT: "#fdba74",
    CellClassification.SOLID: "#94a3b8",
}


class CellColorMode(str, Enum):
    """Cell quantity mapped to rectangle fill color."""

    NONE = "none"
    LEVEL = "level"
    CLASSIFICATION = "classification"
    FLUID_FRACTION = "fluid_fraction"


@dataclass(frozen=True, slots=True)
class SvgOptions:
    """Rendering choices that do not change the represented geometry."""

    width: int = 1000
    margin: float = 24.0
    color_by: CellColorMode = CellColorMode.LEVEL
    show_obstacle: bool = True
    show_cut_boundaries: bool = True
    show_snapped_points: bool = False
    show_normals: bool = False
    show_legend: bool = True
    normal_scale: float = 0.35

    def __post_init__(self) -> None:
        if not isinstance(self.width, int) or isinstance(self.width, bool):
            raise TypeError("SVG width must be an integer")
        if self.width < 200:
            raise ValueError("SVG width must be at least 200 pixels")
        if self.margin < 0.0:
            raise ValueError("SVG margin must be non-negative")
        if self.normal_scale <= 0.0:
            raise ValueError("normal_scale must be positive")
        try:
            mode = CellColorMode(self.color_by)
        except ValueError as error:
            raise ValueError(f"unsupported cell color mode: {self.color_by}") from error
        object.__setattr__(self, "color_by", mode)


@dataclass(frozen=True, slots=True)
class _Canvas2D:
    origin: Point2D
    extent: Point2D
    width: float
    height: float
    margin: float

    @property
    def plot_width(self) -> float:
        return self.width - 2.0 * self.margin

    @property
    def plot_height(self) -> float:
        return self.height - 2.0 * self.margin

    def screen(self, point: Point2D) -> Point2D:
        x, y = point
        return (
            self.margin + self.plot_width * (x - self.origin[0]) / self.extent[0],
            self.margin
            + self.plot_height * (1.0 - (y - self.origin[1]) / self.extent[1]),
        )


def _make_canvas(tree: AdaptiveTree, options: SvgOptions) -> _Canvas2D:
    plot_width = options.width - 2.0 * options.margin
    if plot_width <= 0.0:
        raise ValueError("SVG margin leaves no room for the plot")
    plot_height = plot_width * tree.extent[1] / tree.extent[0]
    return _Canvas2D(
        origin=(tree.origin[0], tree.origin[1]),
        extent=(tree.extent[0], tree.extent[1]),
        width=float(options.width),
        height=plot_height + 2.0 * options.margin,
        margin=options.margin,
    )


def _snapped_lookup(
    tree: AdaptiveTree, snapped: SnappedGeometry2D | None
) -> dict[Cell, SnappedCell2D]:
    if snapped is None:
        return {}

    lookup: dict[Cell, SnappedCell2D] = {}
    for snapped_cell in snapped.cells:
        if snapped_cell.cell in lookup:
            raise ValueError("snapped geometry contains a duplicate cell")
        lookup[snapped_cell.cell] = snapped_cell

    if set(lookup) != set(tree.leaves):
        raise ValueError(
            "snapped geometry is stale or belongs to a different adaptive tree"
        )
    return lookup


def _interpolate_color(start: str, end: str, fraction: float) -> str:
    fraction = max(0.0, min(1.0, fraction))
    start_channels = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
    end_channels = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
    channels = tuple(
        round(first + fraction * (second - first))
        for first, second in zip(start_channels, end_channels, strict=True)
    )
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def _cell_fill(
    cell: Cell,
    snapped_cell: SnappedCell2D | None,
    mode: CellColorMode,
) -> str:
    if mode is CellColorMode.NONE:
        return "none"
    if mode is CellColorMode.LEVEL:
        return _LEVEL_COLORS[min(cell.level, len(_LEVEL_COLORS) - 1)]
    if snapped_cell is None:
        raise ValueError(f"{mode.value} coloring requires snapped geometry")
    if mode is CellColorMode.CLASSIFICATION:
        return _CLASSIFICATION_COLORS[snapped_cell.classification]
    return _interpolate_color("#64748b", "#ffffff", snapped_cell.fluid_fraction)


def _line_parts(geometry: BaseGeometry) -> tuple[tuple[Point2D, ...], ...]:
    if geometry.is_empty:
        return ()
    if isinstance(geometry, (LineString, LinearRing)):
        return (tuple((float(x), float(y)) for x, y in geometry.coords),)
    if hasattr(geometry, "geoms"):
        return tuple(
            part
            for sub_geometry in geometry.geoms
            for part in _line_parts(sub_geometry)
        )
    return ()


def _point_string(canvas: _Canvas2D, points: tuple[Point2D, ...]) -> str:
    return " ".join(
        f"{screen_x:.6f},{screen_y:.6f}"
        for screen_x, screen_y in map(canvas.screen, points)
    )


def _legend_entries(
    tree: AdaptiveTree,
    mode: CellColorMode,
) -> tuple[tuple[str, str], ...]:
    if mode is CellColorMode.LEVEL:
        return tuple(
            (f"level {level}", _LEVEL_COLORS[min(level, len(_LEVEL_COLORS) - 1)])
            for level in sorted({cell.level for cell in tree.leaves})
        )
    if mode is CellColorMode.CLASSIFICATION:
        return tuple(
            (classification.value, _CLASSIFICATION_COLORS[classification])
            for classification in CellClassification
        )
    if mode is CellColorMode.FLUID_FRACTION:
        return (
            ("chi = 0", _interpolate_color("#64748b", "#ffffff", 0.0)),
            ("chi = 0.5", _interpolate_color("#64748b", "#ffffff", 0.5)),
            ("chi = 1", _interpolate_color("#64748b", "#ffffff", 1.0)),
        )
    return ()


def _render_legend(
    canvas: _Canvas2D, entries: tuple[tuple[str, str], ...]
) -> list[str]:
    if not entries:
        return []

    item_height = 20.0
    box_width = 126.0
    box_height = 12.0 + item_height * len(entries)
    x = canvas.width - canvas.margin - box_width - 8.0
    y = canvas.margin + 8.0
    lines = [
        '<g class="legend">',
        f'<rect x="{x:.3f}" y="{y:.3f}" width="{box_width:.3f}" '
        f'height="{box_height:.3f}" rx="4" fill="#ffffff" '
        'fill-opacity="0.92" stroke="#94a3b8" stroke-width="0.8"/>',
    ]
    for index, (label, color) in enumerate(entries):
        item_y = y + 8.0 + index * item_height
        lines.append(
            f'<rect x="{x + 8.0:.3f}" y="{item_y:.3f}" width="14" '
            f'height="14" fill="{color}" stroke="#475569" stroke-width="0.6"/>'
        )
        lines.append(
            f'<text x="{x + 28.0:.3f}" y="{item_y + 11.5:.3f}" '
            f'font-family="sans-serif" font-size="12" fill="#0f172a">'
            f"{escape(label)}</text>"
        )
    lines.append("</g>")
    return lines


def render_geometry_svg(
    tree: AdaptiveTree,
    *,
    obstacle: Obstacle2D | None = None,
    snapped: SnappedGeometry2D | None = None,
    options: SvgOptions | None = None,
) -> str:
    """Render a self-contained SVG of a 2D tree and optional cut-cell geometry.

    Every cell group contains machine-readable ``data-*`` attributes and a tooltip.
    This makes the SVG both a visual diagnostic and a lightweight inspectable export.
    """

    if tree.dimension != 2:
        raise ValueError("SVG geometry rendering currently requires a 2D tree")
    options = SvgOptions() if options is None else options
    canvas = _make_canvas(tree, options)
    snapped_by_cell = _snapped_lookup(tree, snapped)
    if options.color_by in (
        CellColorMode.CLASSIFICATION,
        CellColorMode.FLUID_FRACTION,
    ) and snapped is None:
        raise ValueError(f"{options.color_by.value} coloring requires snapped geometry")

    title = "ai-native-cfd adaptive geometry"
    if obstacle is not None:
        title += f": {obstacle.name}"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {canvas.width:.3f} {canvas.height:.3f}" '
        f'width="{canvas.width:.0f}" height="{canvas.height:.0f}">',
        f"<title>{escape(title)}</title>",
        "<desc>Adaptive Cartesian leaves with optional embedded-boundary data.</desc>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
    ]
    if options.show_normals:
        lines.extend(
            [
                "<defs>",
                '<marker id="normal-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
                'markerWidth="5" markerHeight="5" orient="auto-start-reverse">',
                '<path d="M 0 0 L 10 5 L 0 10 z" fill="#dc2626"/>',
                "</marker>",
                "</defs>",
            ]
        )

    lines.append('<g class="cells">')
    for cell in tree.leaves:
        snapped_cell = snapped_by_cell.get(cell)
        fill = _cell_fill(cell, snapped_cell, options.color_by)
        (lower_x, upper_x), (lower_y, upper_y) = tree.physical_bounds(cell)
        screen_x, screen_y = canvas.screen((lower_x, upper_y))
        cell_width = canvas.plot_width * (upper_x - lower_x) / tree.extent[0]
        cell_height = canvas.plot_height * (upper_y - lower_y) / tree.extent[1]
        classification = (
            snapped_cell.classification.value if snapped_cell is not None else "unknown"
        )
        fluid_fraction = (
            f"{snapped_cell.fluid_fraction:.12g}"
            if snapped_cell is not None
            else "unknown"
        )
        tooltip = (
            f"{cell.stable_id}; level={cell.level}; "
            f"classification={classification}; fluid_fraction={fluid_fraction}"
        )
        lines.extend(
            [
                f'<g class="cell" data-cell-id="{escape(cell.stable_id)}" '
                f'data-level="{cell.level}" data-classification="{classification}" '
                f'data-fluid-fraction="{fluid_fraction}">',
                f"<title>{escape(tooltip)}</title>",
                f'<rect x="{screen_x:.6f}" y="{screen_y:.6f}" '
                f'width="{cell_width:.6f}" height="{cell_height:.6f}" '
                f'fill="{fill}" stroke="#475569" stroke-width="0.7"/>',
                "</g>",
            ]
        )
    lines.append("</g>")

    if obstacle is not None and options.show_obstacle:
        boundary = obstacle.vertices + obstacle.vertices[:1]
        lines.extend(
            [
                '<g class="obstacle">',
                f'<polygon points="{_point_string(canvas, boundary)}" '
                'fill="#334155" fill-opacity="0.22" stroke="#0f172a" '
                'stroke-width="2.4" stroke-linejoin="round"/>',
                "</g>",
            ]
        )

    if snapped is not None and options.show_cut_boundaries:
        lines.append('<g class="cut-boundaries">')
        for snapped_cell in snapped.cut_cells:
            for points in _line_parts(snapped_cell.boundary_fragment):
                lines.append(
                    f'<polyline points="{_point_string(canvas, points)}" fill="none" '
                    'stroke="#c2410c" stroke-width="1.5" stroke-linecap="round"/>'
                )
        lines.append("</g>")

    if snapped is not None and options.show_snapped_points:
        lines.append('<g class="snapped-points" fill="#be123c">')
        points = sorted(
            {
                point
                for snapped_cell in snapped.cut_cells
                for point in snapped_cell.snapped_boundary_points
            }
        )
        for point in points:
            screen_x, screen_y = canvas.screen(point)
            lines.append(
                f'<circle cx="{screen_x:.6f}" cy="{screen_y:.6f}" r="2.4"/>'
            )
        lines.append("</g>")

    if snapped is not None and options.show_normals:
        lines.append(
            '<g class="boundary-normals" fill="none" stroke="#dc2626" '
            'stroke-width="1.2" marker-end="url(#normal-arrow)">'
        )
        for snapped_cell in snapped.cut_cells:
            start = snapped_cell.nearest_boundary_point
            length = options.normal_scale * min(tree.cell_size(snapped_cell.cell))
            end = (
                start[0] + length * snapped_cell.boundary_normal[0],
                start[1] + length * snapped_cell.boundary_normal[1],
            )
            start_x, start_y = canvas.screen(start)
            end_x, end_y = canvas.screen(end)
            lines.append(
                f'<line x1="{start_x:.6f}" y1="{start_y:.6f}" '
                f'x2="{end_x:.6f}" y2="{end_y:.6f}"/>'
            )
        lines.append("</g>")

    if options.show_legend:
        lines.extend(_render_legend(canvas, _legend_entries(tree, options.color_by)))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_geometry_svg(
    path: str | Path,
    tree: AdaptiveTree,
    *,
    obstacle: Obstacle2D | None = None,
    snapped: SnappedGeometry2D | None = None,
    options: SvgOptions | None = None,
) -> Path:
    """Render and write an SVG, returning the output path."""

    output_path = Path(path)
    output_path.write_text(
        render_geometry_svg(
            tree, obstacle=obstacle, snapped=snapped, options=options
        ),
        encoding="utf-8",
    )
    return output_path
