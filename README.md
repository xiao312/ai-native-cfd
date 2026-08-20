# ai-native-cfd

`ai-native-cfd` is a learning-oriented research project for hierarchical adaptive
representations and AI-enhanced numerical methods for computational fluid
dynamics.

The project is deliberately starting below the CFD-solver level. Its first task is
to represent an adaptive Cartesian mesh clearly and correctly in Python:

- a binary tree in one dimension;
- a quadtree in two dimensions;
- an octree in three dimensions.

The same dimension-independent code handles all three cases.

## Current scope

The package currently provides two small data structures:

- `Cell` identifies one cell by its refinement level and integer grid index;
- `AdaptiveTree` owns the current leaf cells and implements refinement,
  coarsening, face-neighbour queries, point lookup, and optional 2:1 balancing.

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

## Near-term roadmap

1. Add conservative scalar-field transfer during refinement and coarsening.
2. Build physical face segments across coarse-fine interfaces.
3. Implement a manufactured scalar advection-diffusion problem.
4. Add offline OpenFOAM snapshot readers and projection utilities.
5. Introduce deterministic, then learned, refinement policies.
