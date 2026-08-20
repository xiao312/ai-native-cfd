"""Inspect AMR hierarchy, faces, ghosts, divergence, and one reflux correction."""

from __future__ import annotations

from collections import Counter

import numpy as np

from aicfd import (
    AdaptiveTree,
    AMRHierarchy,
    Cell,
    CellField,
    FaceField,
    FaceTopology,
    FieldLocation,
    FieldSpec,
    FluxRegister,
    GhostTopology,
    TreeLayout,
    ValueRepresentation,
    field_total,
    flux_divergence,
)


def inspect_adaptive_quadtree() -> None:
    """Build a seven-leaf quadtree and print its numerical topology."""

    tree = AdaptiveTree(2)
    tree.refine(tree.root)
    tree.refine(Cell(1, (0, 0)))
    layout = TreeLayout.from_tree(tree)
    hierarchy = AMRHierarchy(layout)
    faces = FaceTopology(layout)
    ghosts = GhostTopology(faces)

    # A constant physical flux vector (1, 0) has normal component dot(F, n)=n_x.
    # Its divergence must be zero even where one coarse face becomes two segments.
    normal_flux = np.array([face.normal[0] for face in faces.faces])
    flux = FaceField(
        FieldSpec(
            "constant_x_flux",
            location=FieldLocation.FACE,
            value_representation=ValueRepresentation.FACE_AVERAGE,
        ),
        faces,
        normal_flux,
    )
    divergence = flux_divergence(flux)

    print("2D adaptive hierarchy")
    for level in hierarchy.levels:
        print(
            f"  level {level.number}: nodes={len(level.cells)}, "
            f"active={len(level.active_cells)}, refined={len(level.refined_cells)}"
        )
    print(
        f"  faces: total={len(faces)}, boundary={len(faces.boundary_faces)}, "
        f"coarse-fine segments={len(faces.coarse_fine_faces)}"
    )
    ghost_counts = Counter(slot.source_kind.value for slot in ghosts.slots)
    print(f"  ghost recipes: {dict(sorted(ghost_counts.items()))}")
    print(f"  max |divergence(constant flux)|: {np.max(np.abs(divergence)):.3e}")


def demonstrate_refluxing() -> None:
    """Correct a deliberately mismatched one-dimensional interface flux."""

    tree = AdaptiveTree(1)
    tree.refine(tree.root)
    tree.refine(Cell(1, (0,)))
    layout = TreeLayout.from_tree(tree)
    faces = FaceTopology(layout)
    interface = faces.coarse_fine_faces[0]
    interface_row = faces.index(interface)
    flux_spec = FieldSpec(
        "q_flux",
        location=FieldLocation.FACE,
        value_representation=ValueRepresentation.FACE_AVERAGE,
    )

    coarse_values = np.zeros(len(faces))
    fine_values = np.zeros(len(faces))
    coarse_values[interface_row] = 2.0
    fine_values[interface_row] = 3.0
    register = FluxRegister(faces)
    register.accumulate_coarse(
        FaceField(flux_spec, faces, coarse_values),
        delta_t=1.0,
    )
    # Two half steps illustrate fine-level subcycling.
    fine_flux = FaceField(flux_spec, faces, fine_values)
    register.accumulate_fine(fine_flux, delta_t=0.5)
    register.accumulate_fine(fine_flux, delta_t=0.5)

    # Mimic the provisional updates: the fine owner loses 3 units while the
    # coarse neighbor gains only 2, leaving a global deficit of 1.
    values = np.zeros(len(layout))
    values[layout.index(interface.owner)] = -3.0 / layout.cell_measure(interface.owner)
    coarse_cell = interface.coarse_cell
    assert coarse_cell is not None
    values[layout.index(coarse_cell)] = 2.0 / layout.cell_measure(coarse_cell)
    provisional = CellField(FieldSpec("q"), layout, values)
    corrected = register.reflux(provisional)

    print("1D reflux demonstration")
    print(f"  composite total before reflux: {field_total(provisional):.6f}")
    print(f"  fine-minus-coarse flux mismatch: {register.mismatch[0, 0]:.6f}")
    print(f"  composite total after reflux:  {field_total(corrected):.6f}")


def main() -> None:
    inspect_adaptive_quadtree()
    demonstrate_refluxing()


if __name__ == "__main__":
    main()
