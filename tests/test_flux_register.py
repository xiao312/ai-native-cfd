import numpy as np
import pytest

from aicfd import (
    AdaptiveTree,
    Cell,
    CellField,
    FaceField,
    FaceTopology,
    FieldLocation,
    FieldSpec,
    FluxRegister,
    TreeLayout,
    ValueRepresentation,
    field_total,
)


def _one_dimensional_interface():
    tree = AdaptiveTree(1)
    tree.refine(tree.root)
    tree.refine(Cell(1, (0,)))
    layout = TreeLayout.from_tree(tree)
    topology = FaceTopology(layout)
    return layout, topology, topology.coarse_fine_faces[0]


def _face_flux(topology: FaceTopology, interface_value: float) -> FaceField:
    values = np.zeros(len(topology))
    values[topology.index(topology.coarse_fine_faces[0])] = interface_value
    return FaceField(
        FieldSpec(
            "q_flux",
            location=FieldLocation.FACE,
            value_representation=ValueRepresentation.FACE_AVERAGE,
        ),
        topology,
        values,
    )


def test_reflux_restores_composite_conservation_after_subcycling() -> None:
    layout, topology, interface = _one_dimensional_interface()
    register = FluxRegister(topology)
    register.accumulate_coarse(_face_flux(topology, 2.0), delta_t=1.0)
    register.accumulate_fine(_face_flux(topology, 3.0), delta_t=0.5)
    register.accumulate_fine(_face_flux(topology, 3.0), delta_t=0.5)

    # The fine owner lost three units while the coarse neighbor gained only two.
    # Values below are cell averages, so divide those amounts by cell lengths.
    uncorrected_values = np.zeros(len(layout))
    uncorrected_values[layout.index(interface.owner)] = -3.0 / layout.cell_measure(
        interface.owner
    )
    coarse = interface.coarse_cell
    assert coarse is not None
    uncorrected_values[layout.index(coarse)] = 2.0 / layout.cell_measure(coarse)
    uncorrected = CellField(FieldSpec("q"), layout, uncorrected_values)

    corrected = register.reflux(uncorrected, clear_after=True)

    assert field_total(uncorrected) == pytest.approx(-1.0)
    assert field_total(corrected) == pytest.approx(0.0)
    assert corrected.at(interface.owner) == pytest.approx(
        uncorrected.at(interface.owner)
    )
    assert corrected.at(coarse) == pytest.approx(6.0)
    assert register.is_empty


def test_register_stores_one_row_per_coarse_fine_segment() -> None:
    tree = AdaptiveTree(2)
    tree.refine(tree.root)
    tree.refine(Cell(1, (0, 0)))
    topology = FaceTopology(TreeLayout.from_tree(tree))

    register = FluxRegister(topology, n_components=3)

    assert len(register.interface_faces) == 4
    assert register.coarse_integrals.shape == (4, 3)
    assert register.fine_integrals.shape == (4, 3)
    assert register.mismatch.shape == (4, 3)


def test_register_rejects_flux_from_another_topology() -> None:
    _, topology, _ = _one_dimensional_interface()
    other_tree = AdaptiveTree(1)
    other_tree.refine(other_tree.root)
    other_topology = FaceTopology(TreeLayout.from_tree(other_tree))
    other_flux = FaceField(
        FieldSpec(
            "flux",
            location=FieldLocation.FACE,
            value_representation=ValueRepresentation.FACE_AVERAGE,
        ),
        other_topology,
        np.zeros(len(other_topology)),
    )

    with pytest.raises(ValueError, match="different topology"):
        FluxRegister(topology).accumulate_coarse(other_flux, 1.0)


def test_flux_register_contains_only_coarse_fine_faces() -> None:
    _, topology, _ = _one_dimensional_interface()
    register = FluxRegister(topology)

    with pytest.raises(KeyError, match="only coarse-fine"):
        register.row(topology.boundary_faces[0])
