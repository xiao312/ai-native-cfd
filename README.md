# ai-native-cfd

`ai-native-cfd` is a learning-oriented research project for hierarchical adaptive
representations and AI-enhanced numerical methods for computational fluid
dynamics.

The project is deliberately starting below the CFD-solver level. Its first tasks
are to represent an adaptive Cartesian mesh clearly and correctly in Python and to
embed two-dimensional obstacle geometry in that hierarchy:

- a binary tree in one dimension;
- a quadtree in two dimensions;
- an octree in three dimensions.

The same dimension-independent code handles all three cases.

## Current scope

The package currently provides seven small layers:

- `Cell` identifies one cell by its refinement level and integer grid index;
- `AdaptiveTree` owns the current leaf cells and implements refinement,
  coarsening, face-neighbour queries, point lookup, and optional 2:1 balancing.
- `Obstacle2D` plus the geometry-adaptation utilities refine around a circle,
  NACA four-digit airfoil, or any valid closed polyline, then conservatively clip
  cut cells onto that boundary.
- `TreeLayout`, `FieldSpec`, `CellField`, and `State` bind dense NumPy field arrays
  to an immutable leaf snapshot without putting physical values inside `Cell`.
- Conservative transfer and checkpoint utilities remap cell averages or integrals
  during refinement/coarsening and store topology before field data.
- `aicfd.amr` supplies value-range, neighbour-jump, and first-order wavelet-detail
  indicators behind a separate policy for hysteresis, buffers, geometry floors,
  2:1 balance, and cell budgets.
- `aicfd.visualization` renders those discretizations as self-contained SVG files,
  colored by tree level, cell classification, or fluid fraction.

This is an educational reference implementation. It favors direct algorithms and
clear invariants over performance. In particular, neighbour searches are currently
quadratic in the number of leaf cells. That is acceptable for tiny verification
problems and gives us a trustworthy baseline before introducing optimized data
structures.

## Quick example

```python
from aicfd import AdaptiveTree

mesh = AdaptiveTree(dimension=2)

# Refining a 2D cell replaces it with four children.
level_one = mesh.refine(mesh.root)

# Refine the lower-left child once more.
mesh.refine(level_one[0])

# Enforce that face-neighbouring cells differ by at most one level.
mesh.balance()

for cell in mesh.leaves:
    print(cell.stable_id, mesh.physical_bounds(cell))
```

## Development

The package requires Python 3.10 or newer. With
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --dev
uv run pytest
uv run ruff check .
```

The tests are intentionally small and do not launch CFD or OpenFOAM simulations.
OpenFOAM will initially be treated as an external source of reference solutions,
not as the owner of this package's mesh representation.

For a beginner-oriented explanation of the tree algorithms, see
[`docs/learning/01-adaptive-cartesian-trees.md`](docs/learning/01-adaptive-cartesian-trees.md).
The next guide explains
[`geometry-driven AMR and snapping`](docs/learning/02-geometry-driven-amr.md).
The third explains the
[`SVG visualization layers`](docs/learning/03-visualizing-geometry.md), and the
fourth introduces
[`physical fields and solution-driven AMR`](docs/learning/04-physical-fields-and-solution-amr.md).
The research notes record what we found about the historical
[`OpenFOAM dolphin mesh`](docs/research/01-openfoam-dolphin.md), the deferred
[`cut-cell quality framework`](docs/research/02-cut-cell-quality.md), and the
recommended design for
[`physical fields and solution-driven AMR`](docs/research/03-field-storage-and-solution-amr.md).

The geometry-only example produces a small SVG preview and runs no CFD solver:

```bash
uv run python examples/geometry_amr.py --shape circle --output circle.svg
uv run python examples/geometry_amr.py --shape naca0012 --output naca0012.svg
uv run python examples/geometry_amr.py --shape naca0012 --show-points \
  --show-normals --color-by fluid_fraction --output naca0012-debug.svg
```

The manufactured-field example performs one tiny AMR transaction, checks the
conserved scalar integral, and also runs no CFD solver:

```bash
uv run python examples/field_amr.py
```

## Near-term roadmap

1. Build physical face segments across coarse-fine and cut-cell interfaces.
2. Add limited second-order prolongation and local gradient reconstruction.
3. Add scalar-field visualization/export on adaptive leaves.
4. Implement a manufactured scalar advection-diffusion problem and refluxing.
5. Add narrow-gap detection and the deferred cut-cell quality diagnostics.
6. Add offline OpenFOAM snapshot readers and projection utilities.
7. Introduce learned refinement policies behind deterministic safety controls.
