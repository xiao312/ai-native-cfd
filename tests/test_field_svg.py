from xml.etree import ElementTree

import numpy as np
import pytest

from aicfd import AdaptiveTree, CellField, FieldSpec, TreeLayout
from aicfd.visualization import (
    FieldSvgOptions,
    render_field_svg,
    write_field_svg,
)

_SVG_NAMESPACE = {"svg": "http://www.w3.org/2000/svg"}


def _layout() -> TreeLayout:
    tree = AdaptiveTree(2)
    tree.refine(tree.root)
    return TreeLayout.from_tree(tree)


def test_field_svg_contains_one_value_carrying_group_per_cell() -> None:
    layout = _layout()
    field = CellField(FieldSpec("p", unit="Pa"), layout, [-2.0, -1.0, 1.0, 2.0])

    root = ElementTree.fromstring(
        render_field_svg(field, options=FieldSvgOptions(symmetric=True))
    )

    cells = root.findall(".//svg:g[@class='field-cell']", _SVG_NAMESPACE)
    assert len(cells) == len(layout)
    assert {float(cell.attrib["data-value"]) for cell in cells} == {
        -2.0,
        -1.0,
        1.0,
        2.0,
    }
    assert root.find(".//svg:g[@class='field-legend']", _SVG_NAMESPACE) is not None


def test_field_svg_can_render_vector_magnitude_and_write_file(tmp_path) -> None:
    layout = _layout()
    field = CellField(
        FieldSpec("U", component_names=("x", "y"), unit="m/s"),
        layout,
        [[3.0, 4.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
    )
    output = tmp_path / "speed.svg"

    returned = write_field_svg(output, field, component="magnitude")
    root = ElementTree.fromstring(output.read_text(encoding="utf-8"))

    assert returned == output
    values = root.findall(".//svg:g[@class='field-cell']", _SVG_NAMESPACE)
    assert float(values[0].attrib["data-value"]) == pytest.approx(5.0)


def test_field_svg_rejects_magnitude_for_scalar_fields() -> None:
    layout = _layout()
    field = CellField(FieldSpec("phi"), layout, np.ones(len(layout)))

    with pytest.raises(ValueError, match="vector"):
        render_field_svg(field, component="magnitude")
