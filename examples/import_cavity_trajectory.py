"""Convert the OpenFOAM cavity64 run into a compact quadtree trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aicfd import CellField, FieldSpec, write_trajectory
from aicfd.io import import_openfoam_cartesian_2d
from aicfd.visualization import FieldSvgOptions, write_field_svg


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, help="completed OpenFOAM cavity64 case")
    parser.add_argument("output", type=Path, help="new trajectory output directory")
    parser.add_argument(
        "--previews",
        type=Path,
        help="optional directory for SVG previews at the first, middle, and last time",
    )
    return parser.parse_args()


def _write_previews(trajectory, output_directory: Path) -> tuple[Path, ...]:
    speed = np.linalg.norm(trajectory["U"].values, axis=2)
    speed_maximum = max(float(np.max(speed)), 1.0e-12)
    pressure = trajectory["p"].values
    centered_pressure = pressure - np.mean(pressure, axis=1, keepdims=True)
    pressure_bound = max(
        float(np.quantile(np.abs(centered_pressure), 0.99)),
        1.0e-12,
    )
    pressure_spec = FieldSpec(
        "p_centered",
        unit=trajectory["p"].spec.unit,
        unit_dimensions=trajectory["p"].spec.unit_dimensions,
    )

    outputs: list[Path] = []
    for frame in (0, len(trajectory) // 2, len(trajectory) - 1):
        state = trajectory.state(frame)
        time_label = f"{state.time:.5f}".rstrip("0").rstrip(".")
        safe_time = time_label.replace(".", "p")
        outputs.append(
            write_field_svg(
                output_directory / f"speed-t{safe_time}.svg",
                state["U"],
                component="magnitude",
                options=FieldSvgOptions(
                    show_grid=False,
                    value_min=0.0,
                    value_max=speed_maximum,
                    title=f"cavity64 speed at t={time_label} s",
                ),
            )
        )
        centered_field = CellField(
            pressure_spec,
            trajectory.layout,
            centered_pressure[frame],
        )
        outputs.append(
            write_field_svg(
                output_directory / f"pressure-t{safe_time}.svg",
                centered_field,
                options=FieldSvgOptions(
                    show_grid=False,
                    symmetric=True,
                    value_min=-pressure_bound,
                    value_max=pressure_bound,
                    title=(
                        "cavity64 centered pressure (99% color range) "
                        f"at t={time_label} s"
                    ),
                ),
            )
        )
    return tuple(outputs)


def main() -> None:
    arguments = _arguments()
    trajectory, report = import_openfoam_cartesian_2d(
        arguments.case,
        level=6,
        origin=(0.0, 0.0),
        extent=(0.1, 0.1),
        delta_t=0.00125,
        openfoam_version="7",
    )
    if len(trajectory) != 401:
        raise ValueError(f"expected 401 cavity frames, found {len(trajectory)}")
    if not np.isclose(report.first_time, 0.0) or not np.isclose(report.last_time, 0.5):
        raise ValueError(
            "expected cavity time range 0 -> 0.5 s, found "
            f"{report.first_time:g} -> {report.last_time:g} s"
        )
    write_trajectory(arguments.output, trajectory)
    previews = (
        _write_previews(trajectory, arguments.previews)
        if arguments.previews is not None
        else ()
    )

    print(f"frames: {report.time_frames}")
    print(f"cells: {report.source_cells}")
    print(f"time range: {report.first_time:g} -> {report.last_time:g} s")
    print(f"maximum cell-centre error: {report.maximum_center_error:.3e} m")
    print(
        f"maximum out-of-plane velocity: {report.maximum_out_of_plane_velocity:.3e} m/s"
    )
    print(f"trajectory: {arguments.output}")
    for preview in previews:
        print(f"preview: {preview}")


if __name__ == "__main__":
    main()
