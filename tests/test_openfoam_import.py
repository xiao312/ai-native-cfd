from pathlib import Path

import numpy as np
import pytest

from aicfd import Cell
from aicfd.io import import_openfoam_cartesian_2d, read_foam_field


def _write_field(
    path: Path,
    *,
    name: str,
    field_class: str,
    dimensions: str,
    values: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
        FoamFile
        {{
            format ascii;
            class {field_class};
            object {name};
        }}
        dimensions [{dimensions}];
        internalField {values};
        boundaryField {{}}
        """,
        encoding="utf-8",
    )


def _nonuniform_scalar(values: list[float]) -> str:
    body = "\n".join(str(value) for value in values)
    return f"nonuniform List<scalar> {len(values)}\n(\n{body}\n)"


def _nonuniform_vector(values: list[tuple[float, float, float]]) -> str:
    body = "\n".join(f"({x} {y} {z})" for x, y, z in values)
    return f"nonuniform List<vector> {len(values)}\n(\n{body}\n)"


def test_read_foam_field_handles_uniform_vector_and_nonuniform_scalar(
    tmp_path,
) -> None:
    velocity_path = tmp_path / "U"
    pressure_path = tmp_path / "p"
    _write_field(
        velocity_path,
        name="U",
        field_class="volVectorField",
        dimensions="0 1 -1 0 0 0 0",
        values="uniform (1 2 0)",
    )
    _write_field(
        pressure_path,
        name="p",
        field_class="volScalarField",
        dimensions="0 2 -2 0 0 0 0",
        values=_nonuniform_scalar([1.0, 2.0, 3.0]),
    )

    velocity = read_foam_field(velocity_path)
    pressure = read_foam_field(pressure_path)

    assert velocity.uniform
    np.testing.assert_array_equal(
        velocity.values_for_count(2),
        [[1.0, 2.0, 0.0], [1.0, 2.0, 0.0]],
    )
    assert not pressure.uniform
    np.testing.assert_array_equal(pressure.values_for_count(3), [1.0, 2.0, 3.0])


def test_import_openfoam_cartesian_case_reorders_cells_into_tree_layout(
    tmp_path,
) -> None:
    # Deliberately use an order unrelated to TreeLayout's Morton ordering.
    source_indices = [(1, 0), (0, 1), (0, 0), (1, 1)]
    centers = [((i + 0.5) / 2, (j + 0.5) / 2, 0.05) for i, j in source_indices]
    _write_field(
        tmp_path / "0" / "C",
        name="C",
        field_class="volVectorField",
        dimensions="0 1 0 0 0 0 0",
        values=_nonuniform_vector(centers),
    )

    expected_by_cell: dict[Cell, float] = {}
    for time_name, offset in (("0", 0.0), ("0.1", 100.0)):
        velocities = []
        pressures = []
        for i, j in source_indices:
            value = offset + 10.0 * i + j
            velocities.append((value, -value, 0.0))
            pressures.append(value + 0.5)
            if time_name == "0.1":
                expected_by_cell[Cell(1, (i, j))] = value
        _write_field(
            tmp_path / time_name / "U",
            name="U",
            field_class="volVectorField",
            dimensions="0 1 -1 0 0 0 0",
            values=_nonuniform_vector(velocities),
        )
        _write_field(
            tmp_path / time_name / "p",
            name="p",
            field_class="volScalarField",
            dimensions="0 2 -2 0 0 0 0",
            values=_nonuniform_scalar(pressures),
        )

    trajectory, report = import_openfoam_cartesian_2d(
        tmp_path,
        level=1,
        origin=(0.0, 0.0),
        extent=(1.0, 1.0),
        delta_t=0.1,
        openfoam_version="test",
    )

    assert len(trajectory) == 2
    assert report.source_cells == 4
    assert report.maximum_center_error == pytest.approx(0.0)
    assert report.maximum_out_of_plane_velocity == pytest.approx(0.0)
    state = trajectory.state(1)
    for cell, expected in expected_by_cell.items():
        assert state["U"].at(cell)[0] == pytest.approx(expected)


def test_import_rejects_centres_that_do_not_match_the_requested_grid(
    tmp_path,
) -> None:
    centers = [
        (0.3, 0.25, 0.05),
        (0.75, 0.25, 0.05),
        (0.25, 0.75, 0.05),
        (0.75, 0.75, 0.05),
    ]
    _write_field(
        tmp_path / "0" / "C",
        name="C",
        field_class="volVectorField",
        dimensions="0 1 0 0 0 0 0",
        values=_nonuniform_vector(centers),
    )
    _write_field(
        tmp_path / "0" / "U",
        name="U",
        field_class="volVectorField",
        dimensions="0 1 -1 0 0 0 0",
        values="uniform (0 0 0)",
    )
    _write_field(
        tmp_path / "0" / "p",
        name="p",
        field_class="volScalarField",
        dimensions="0 2 -2 0 0 0 0",
        values="uniform 0",
    )

    with pytest.raises(ValueError, match="not the requested Cartesian"):
        import_openfoam_cartesian_2d(
            tmp_path,
            level=1,
            origin=(0.0, 0.0),
            extent=(1.0, 1.0),
        )
