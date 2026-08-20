"""Pure-Python SVG rendering for cell-centred physical fields."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from math import isfinite
from pathlib import Path

import numpy as np

from aicfd.fields import CellField

_DEFAULT_PALETTE = (
    "#440154",
    "#3b528b",
    "#21918c",
    "#5ec962",
    "#fde725",
)


@dataclass(frozen=True, slots=True)
class FieldSvgOptions:
    """Rendering options for one scalar component or vector magnitude."""

    width: int = 1000
    margin: float = 24.0
    show_grid: bool = True
    show_legend: bool = True
    symmetric: bool = False
    value_min: float | None = None
    value_max: float | None = None
    title: str | None = None
    palette: tuple[str, ...] = _DEFAULT_PALETTE

    def __post_init__(self) -> None:
        if not isinstance(self.width, int) or isinstance(self.width, bool):
            raise TypeError("SVG width must be an integer")
        if self.width < 200:
            raise ValueError("SVG width must be at least 200 pixels")
        if self.margin < 0.0:
            raise ValueError("SVG margin must be non-negative")
        if len(self.palette) < 2:
            raise ValueError("field palette needs at least two colors")
        for color in self.palette:
            if not isinstance(color, str) or not re_full_hex_color(color):
                raise ValueError(f"invalid SVG palette color {color!r}")
        for name, value in (
            ("value_min", self.value_min),
            ("value_max", self.value_max),
        ):
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if (
            self.value_min is not None
            and self.value_max is not None
            and self.value_min >= self.value_max
        ):
            raise ValueError("value_min must be smaller than value_max")


def re_full_hex_color(value: str) -> bool:
    """Return whether ``value`` is a six-digit SVG hex color."""

    return (
        len(value) == 7
        and value[0] == "#"
        and all(character in "0123456789abcdefABCDEF" for character in value[1:])
    )


def _interpolate_color(start: str, end: str, fraction: float) -> str:
    fraction = max(0.0, min(1.0, fraction))
    start_channels = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
    end_channels = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
    channels = tuple(
        round(first + fraction * (second - first))
        for first, second in zip(start_channels, end_channels, strict=True)
    )
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def _palette_color(palette: tuple[str, ...], fraction: float) -> str:
    fraction = max(0.0, min(1.0, fraction))
    scaled = fraction * (len(palette) - 1)
    lower = min(int(scaled), len(palette) - 2)
    return _interpolate_color(palette[lower], palette[lower + 1], scaled - lower)


def _selected_values(
    field: CellField,
    component: int | str,
) -> tuple[np.ndarray, str]:
    if component == "magnitude":
        if field.spec.n_components == 1:
            raise ValueError("magnitude rendering requires a vector field")
        return np.linalg.norm(field.values, axis=1), "magnitude"
    values = np.asarray(field.component(component), dtype=np.float64)
    if isinstance(component, int):
        label = field.spec.component_names[component]
    else:
        label = component
    return values, label


def _value_range(
    values: np.ndarray,
    options: FieldSvgOptions,
) -> tuple[float, float]:
    minimum = float(np.min(values)) if options.value_min is None else options.value_min
    maximum = float(np.max(values)) if options.value_max is None else options.value_max
    if options.symmetric:
        bound = max(abs(minimum), abs(maximum))
        if bound == 0.0:
            bound = 1.0
        minimum, maximum = -bound, bound
    elif minimum == maximum:
        scale = max(abs(minimum), 1.0)
        minimum -= 0.5 * scale
        maximum += 0.5 * scale
    if minimum >= maximum:
        raise ValueError("resolved field color range must have positive width")
    return minimum, maximum


def render_field_svg(
    field: CellField,
    *,
    component: int | str = 0,
    options: FieldSvgOptions | None = None,
) -> str:
    """Render a cell field on its two-dimensional adaptive layout."""

    if field.layout.dimension != 2:
        raise ValueError("SVG field rendering currently requires a 2D layout")
    options = FieldSvgOptions() if options is None else options
    values, component_label = _selected_values(field, component)
    minimum, maximum = _value_range(values, options)

    tree = field.layout.to_tree()
    plot_width = options.width - 2.0 * options.margin
    if plot_width <= 0.0:
        raise ValueError("SVG margin leaves no room for the plot")
    plot_height = plot_width * tree.extent[1] / tree.extent[0]
    legend_height = 54.0 if options.show_legend else 0.0
    canvas_height = plot_height + 2.0 * options.margin + legend_height

    def screen(x: float, y: float) -> tuple[float, float]:
        return (
            options.margin + plot_width * (x - tree.origin[0]) / tree.extent[0],
            options.margin
            + plot_height * (1.0 - (y - tree.origin[1]) / tree.extent[1]),
        )

    title = options.title or f"{field.spec.name}: {component_label}"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {options.width:.3f} {canvas_height:.3f}" '
        f'width="{options.width}" height="{canvas_height:.0f}">',
        f"<title>{escape(title)}</title>",
        "<desc>Cell-centred physical field on an adaptive Cartesian layout.</desc>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
    ]
    if options.show_legend:
        lines.extend(
            [
                "<defs>",
                '<linearGradient id="field-colorbar" x1="0%" x2="100%">',
                *[
                    f'<stop offset="{100.0 * index / (len(options.palette) - 1):.3f}%" '
                    f'stop-color="{color}"/>'
                    for index, color in enumerate(options.palette)
                ],
                "</linearGradient>",
                "</defs>",
            ]
        )

    stroke = "#475569" if options.show_grid else "none"
    stroke_width = "0.35" if options.show_grid else "0"
    lines.append('<g class="field-cells">')
    for row, cell in enumerate(field.layout.cells):
        value = float(values[row])
        fraction = (value - minimum) / (maximum - minimum)
        fill = _palette_color(options.palette, fraction)
        (lower_x, upper_x), (lower_y, upper_y) = tree.physical_bounds(cell)
        screen_x, screen_y = screen(lower_x, upper_y)
        cell_width = plot_width * (upper_x - lower_x) / tree.extent[0]
        cell_height = plot_height * (upper_y - lower_y) / tree.extent[1]
        tooltip = (
            f"{cell.stable_id}; {field.spec.name}.{component_label}={value:.12g} "
            f"{field.spec.unit}"
        )
        lines.extend(
            [
                f'<g class="field-cell" data-cell-id="{escape(cell.stable_id)}" '
                f'data-value="{value:.12g}">',
                f"<title>{escape(tooltip)}</title>",
                f'<rect x="{screen_x:.6f}" y="{screen_y:.6f}" '
                f'width="{cell_width:.6f}" height="{cell_height:.6f}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>',
                "</g>",
            ]
        )
    lines.append("</g>")

    if options.show_legend:
        legend_x = options.margin
        legend_y = options.margin + plot_height + 24.0
        legend_width = plot_width
        lines.extend(
            [
                '<g class="field-legend">',
                f'<rect x="{legend_x:.3f}" y="{legend_y:.3f}" '
                f'width="{legend_width:.3f}" height="14" '
                'fill="url(#field-colorbar)" stroke="#475569" stroke-width="0.6"/>',
                f'<text x="{legend_x:.3f}" y="{legend_y + 31.0:.3f}" '
                'font-family="sans-serif" font-size="12" fill="#0f172a">'
                f"{minimum:.6g}</text>",
                f'<text x="{legend_x + 0.5 * legend_width:.3f}" '
                f'y="{legend_y + 31.0:.3f}" text-anchor="middle" '
                'font-family="sans-serif" font-size="12" fill="#0f172a">'
                f"{escape(field.spec.name)}.{escape(component_label)} "
                f"[{escape(field.spec.unit)}]</text>",
                f'<text x="{legend_x + legend_width:.3f}" '
                f'y="{legend_y + 31.0:.3f}" text-anchor="end" '
                'font-family="sans-serif" font-size="12" fill="#0f172a">'
                f"{maximum:.6g}</text>",
                "</g>",
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_field_svg(
    path: str | Path,
    field: CellField,
    *,
    component: int | str = 0,
    options: FieldSvgOptions | None = None,
) -> Path:
    """Render and write a field SVG, returning the output path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_field_svg(field, component=component, options=options),
        encoding="utf-8",
    )
    return output_path
