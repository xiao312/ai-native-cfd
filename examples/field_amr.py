"""Adapt a tiny quadtree around a manufactured Gaussian scalar field."""

from __future__ import annotations

import argparse
from math import erf, pi, sqrt
from pathlib import Path

from aicfd import (
    AdaptiveTree,
    Cell,
    CellField,
    FieldSpec,
    NeighborJumpIndicator,
    SolutionRefinementPolicy,
    State,
    TreeLayout,
    WaveletDetailIndicator,
    adapt_solution,
    field_total,
    write_checkpoint,
)


def gaussian_cell_average(
    tree: AdaptiveTree,
    cell: Cell,
    *,
    center: tuple[float, float] = (0.5, 0.5),
    width: float = 0.12,
) -> float:
    """Average ``exp(-r^2/width^2)`` exactly over one rectangular cell."""

    (x0, x1), (y0, y1) = tree.physical_bounds(cell)

    def integral(lower: float, upper: float, centre: float) -> float:
        return (
            0.5
            * width
            * sqrt(pi)
            * (erf((upper - centre) / width) - erf((lower - centre) / width))
        )

    area = (x1 - x0) * (y1 - y0)
    return integral(x0, x1, center[0]) * integral(y0, y1, center[1]) / area


def make_state() -> State:
    """Create a small uniform starting grid and a conservative scalar field."""

    tree = AdaptiveTree(2)
    for _ in range(3):
        tree.refine_many(tree.leaves)
    layout = TreeLayout.from_tree(tree)
    values = [gaussian_cell_average(tree, cell) for cell in layout.cells]
    field = CellField(FieldSpec("phi", unit="1"), layout, values)
    return State(layout, [field])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="optional new checkpoint directory; existing paths are not overwritten",
    )
    arguments = parser.parse_args()

    state = make_state()
    before = field_total(state["phi"])
    adapted, report = adapt_solution(
        state,
        [
            NeighborJumpIndicator("phi", scale=0.12),
            WaveletDetailIndicator("phi", scale=0.08),
        ],
        SolutionRefinementPolicy(
            min_level=2,
            max_level=4,
            refine_threshold=1.0,
            coarsen_threshold=0.2,
            buffer_layers=1,
            max_cells=160,
        ),
    )
    after = field_total(adapted["phi"])

    print(f"leaves: {report.leaves_before} -> {report.leaves_after}")
    print(f"indicator refinements: {len(report.indicator_refined_parents)}")
    print(f"coarsened families: {len(report.coarsened_parents)}")
    print(f"budget-skipped refinements: {len(report.budget_skipped_parents)}")
    print(f"maximum normalized score: {report.maximum_score:.6f}")
    print(f"scalar integral: {before:.16e} -> {after:.16e}")

    if arguments.checkpoint is not None:
        output = write_checkpoint(arguments.checkpoint, adapted)
        print(f"checkpoint: {output.resolve()}")


if __name__ == "__main__":
    main()
