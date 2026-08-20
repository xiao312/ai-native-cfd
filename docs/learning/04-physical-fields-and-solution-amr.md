# Physical fields and solution-driven AMR from first principles

The tree implemented in the first three learning notes knows which cells exist,
but it deliberately knows nothing about pressure, velocity, or temperature. This
step adds physical values without putting them inside each `Cell` object.

## 1. A cell identity is not an array row

`Cell(level=2, index=(1, 3))` identifies the same spatial region whenever that
cell exists. An array row such as `values[7]` has meaning only for one particular
leaf ordering.

`TreeLayout` freezes the current ordered leaves:

```python
tree = AdaptiveTree(2)
tree.refine(tree.root)
layout = TreeLayout.from_tree(tree)

row = layout.index(layout.cells[0])
```

It also calculates a topology fingerprint from the dimension, physical domain,
and ordered leaves. Refining the tree creates a different layout and fingerprint.
A field from the old layout cannot be registered accidentally on the new one.

The layout can reconstruct a new mutable tree with `layout.to_tree()`. The
original tree is not stored inside the physical state, which prevents an external
call to `tree.refine()` from silently changing what field rows mean.

## 2. Field metadata carries numerical meaning

The minimal scalar declaration is:

```python
spec = FieldSpec("temperature", unit="K")
temperature = CellField(spec, layout, values)
```

`FieldSpec` records:

- the field and component names;
- location, currently `FieldLocation.CELL`;
- floating-point dtype;
- unit text and optional seven SI dimension powers;
- optional lower and upper physical bounds;
- how values must transfer between resolutions.

The default transfer representation is `CELL_AVERAGE`. For finite volume, this
means

```text
q_i = (1 / V_i) integral_over_cell(q dV).
```

It is not merely a value sampled at the cell centre. The extensive total is

```text
Q = sum_i(q_i * V_i).
```

`CELL_INTEGRAL` instead means that each stored value is already an extensive
amount. `POINT_VALUE` is available for sampled data but has no conservation
guarantee.

Face, node, and embedded-boundary locations are reserved in the enum, but the
first implementation rejects them in `CellField`. We should introduce each one
only when its indexing and transfer rules exist.

## 3. A state is a consistent field registry

```python
state = State(layout, [temperature], time=0.0, step=0)
temperature = state["temperature"]
```

All fields in a `State` must have unique names and the same topology fingerprint.
The registry is read-only, although the NumPy arrays themselves can be updated by
a future numerical operator.

This separation gives us three useful properties:

1. `Cell` remains small and hashable.
2. NumPy evaluates a complete field without Python loops over value objects.
3. topology changes cannot reinterpret an old array silently.

## 4. Conservative refinement and coarsening

`adapt_state_topology()` applies explicit topology requests to a copied tree and
then transfers every registered field.

For a cell average, first-order refinement copies the parent value:

```text
q_child = q_parent.
```

This is only first-order accurate, but it conserves the total because child
volumes partition the parent volume.

Coarsening uses a volume-weighted mean:

```text
q_parent = sum(V_child * q_child) / sum(V_child).
```

Cell integrals are divided by child volume fraction during refinement and summed
during coarsening. Point values are copied down and averaged up without claiming
conservation.

The transfer function accepts a `measure_provider`. Its default is full Cartesian
cell volume. `GeometryLevelFloor.fluid_measure` supplies clipped fluid area for a
2D obstacle, so

```text
V_i = fluid_fraction_i * Cartesian_area_i
```

is used in conservation checks.

The input state is unchanged until the new topology, all transferred fields, and
their conserved totals pass validation. This is what **transactional adaptation**
means in this package.

## 5. Indicators propose; policy decides

An indicator returns one finite, non-negative score per old leaf. It never mutates
the tree.

### Value range

`ValueRangeIndicator` returns one inside a configured interval and zero outside.
It is intentionally simple and useful for testing or tracking a known interface
fraction.

### Neighbour jump

```text
score_i = max_neighbours(abs(q_j - q_i)) / field_scale
```

The scale makes the score dimensionless. A value of two means that the largest
face-neighbour jump is twice the chosen meaningful field scale.

### Wavelet detail

The first wavelet implementation is a Haar-like prediction defect:

1. average the leaves under a cell's parent;
2. predict the leaf by copying that parent average back down;
3. measure the difference from the actual leaf value.

It is zero for a constant field and large where a coarser representation would
lose variation. A minimum starting level or geometry refinement must first expose
the feature; a level-zero root alone has no finer detail to inspect.

## 6. Refinement policy

`SolutionRefinementPolicy` owns decisions that are independent of how a score was
calculated:

- refine and coarsen thresholds;
- different thresholds for hysteresis;
- minimum and maximum levels;
- face-neighbour buffer layers;
- 2:1 balancing;
- a deterministic maximum-cell budget.

Several indicators are combined by their maximum normalized score. A cell is
refined if any important field exceeds its tolerance, while a sibling family is
coarsened only when all its children are below the coarsening threshold.

`GeometryLevelFloor` combines this with an obstacle policy. The effective target
is the maximum level requested by geometry, solution, and the global minimum.
Solution-based coarsening therefore cannot erase obstacle resolution.

With a cell budget, true tagged cells are considered before buffer cells, then by
descending score and deterministic cell order. Each candidate is tried together
with any balance refinements. A candidate that would exceed the budget is reported
and skipped.

## 7. Checkpoints are not plot files

`write_checkpoint()` creates a new directory and never overwrites an existing
path:

```text
metadata.json   field specs, time, step, topology fingerprint
topology.npz    leaf levels and integer indices in exact array order
fields.npz      numeric field arrays
```

`load_checkpoint()` rebuilds and validates topology before attaching field data.
NumPy loading uses `allow_pickle=False`. This small format is intended for exact
restart and tests; later VTK or OpenFOAM export will be a separate visualization
path.

## 8. Run the manufactured-field example

The example analytically averages a Gaussian over each starting cell, evaluates
neighbour-jump and wavelet scores, adapts once, and checks the scalar integral:

```bash
uv run python examples/field_amr.py
```

Optionally write a checkpoint to a path that does not already exist:

```bash
uv run python examples/field_amr.py --checkpoint artifacts/checkpoints/gaussian
```

It runs no CFD or OpenFOAM solver.

## 9. Deliberate limitations

This is still a correctness reference rather than a production AMR backend:

- neighbour queries remain quadratic;
- prolongation is constant rather than limited second order;
- wavelet detail uses a first-order parent prediction;
- only cell-centred fields are implemented;
- physical face/ghost topology and reflux bookkeeping now live in the next
  learning layer, but boundary-value filling and PDE fluxes are not implemented;
- the checkpoint format is serial and stores flat leaf arrays;
- adaptation performs one solution-requested level per call.

These limitations give the next numerical steps clear, testable boundaries.

Continue with
[`AMR finite-volume data from first principles`](06-amr-finite-volume-data.md).
