# AMR finite-volume data from first principles

The earlier layers answer two questions: which adaptive cells exist, and which
physical values belong to those cells? A finite-volume evolution step needs a
third layer that answers:

```text
which oriented surface carries an interaction between two cells?
```

This implementation adds that layer without changing the primary state model:

```text
active leaves       -> cell-centered state
face segments       -> local flux interactions
hierarchy nodes     -> multilevel structure
ghost slots         -> stencil source recipes
flux register       -> coarse/fine conservation repair
```

It remains a small tree-AMR reference implementation. AMReX and similar libraries
usually group many same-level cells into rectangular patches or boxes for
performance. We group cells by level but do not introduce patch arrays yet.

## 1. Active cells and structural parents are different

Consider a quadtree in which one level-1 cell is refined:

```text
level 0:  root (refined, no direct state row)

level 1:  refined parent + 3 active cells

level 2:  4 active children
```

There are nine hierarchy nodes but only seven active leaves. `AMRHierarchy`
records both sets:

```python
layout = TreeLayout.from_tree(tree)
hierarchy = AMRHierarchy(layout)

hierarchy.at_level(1).active_cells
hierarchy.at_level(1).refined_cells
```

Only `active_cells` have rows in `CellField`. A refined parent is retained to
support parent-child communication; silently storing a second physical state on
it would create two answers for the same region. A future multilevel time
integrator may keep covered coarse states explicitly, but it must define when
those states are synchronized.

## 2. One coarse face becomes several face segments

At a 2:1 interface in two dimensions:

```text
 +-------+-------+
 | fine  |       |
 +-------+ coarse|
 | fine  |       |
 +-------+-------+
```

the shared coarse side is represented by two non-overlapping `FaceSegment`
objects. Each segment connects exactly one fine leaf and one coarse leaf. Their
areas add up to the complete coarse side area.

This choice gives every interaction one unambiguous row. The implementation does
not identify a face by floating-point coordinates. `FaceKey` uses exact dyadic
topology:

```text
axis + refinement level + integer face index
```

`FaceSegment` then supplies its physical data:

- `owner` and optional `neighbor`;
- physical bounds and center;
- area (length in 2D and area in 3D);
- a signed Cartesian unit normal;
- whether the relationship is boundary, same-level, or coarse-fine.

`FaceTopology` derives all Cartesian boundary and interior segments from one
immutable `TreeLayout`. It validates that the segment areas cover both sides of
every active cell exactly.

## 3. Orientation removes flux-sign ambiguity

For an interior segment, the normal points from the geometrically lower cell,
called `owner`, to `neighbor`:

```text
owner ---- normal ----> neighbor
```

For a domain boundary, the normal points out of the domain. A face value is
positive in this normal direction.

`FaceIncidence.sign` is:

```text
+1 for the owner
-1 for the neighbor
```

Therefore the outward integrated flux rate for cell `i` is

\[
\sum_{f\in i} s_{if}\,F_f A_f,
\]

and the divergence is

\[
(\nabla\cdot F)_i
=
\frac{1}{V_i}
\sum_{f\in i}s_{if}\,F_fA_f.
\]

`flux_divergence()` implements this reference calculation. Every interior rate is
added to its owner and subtracted from its neighbor, so internal exchange cancels
in the global volume integral.

## 4. Face values have their own layout and meaning

`FaceField` binds a dense NumPy array to `FaceTopology`, just as `CellField` binds
one to `TreeLayout`:

```python
flux_spec = FieldSpec(
    "mass_flux",
    location=FieldLocation.FACE,
    value_representation=ValueRepresentation.FACE_AVERAGE,
)
flux = FaceField(flux_spec, face_topology, values)
```

Two representations are explicit:

- `FACE_AVERAGE`: the stored normal flux still needs multiplication by face area;
- `FACE_INTEGRAL`: the stored rate already includes face area.

This distinction prevents an easy dimensional mistake. Time is not included in
either representation; an update or flux register multiplies the rate by its time
step.

## 5. Ghost slots are recipes, not extra physical cells

A local stencil asks for a value beyond each side of a cell. `GhostTopology`
creates one `GhostSlot` per cell, axis, and side and classifies its source:

| Source kind | Meaning |
| --- | --- |
| `SAME_LEVEL` | copy/reconstruct from one same-level neighbor |
| `COARSE_PROLONGATION` | interpolate from one coarser neighbor |
| `FINE_RESTRICTION` | combine values from several finer neighbors |
| `PHYSICAL_BOUNDARY` | ask a boundary-condition module |

For example, the coarse cell in the diagram has one ghost slot whose
`source_cells` contains both fine neighbors. Each adjacent fine cell has a slot
whose source is the single coarse cell.

The current object records topology only. It does not invent a boundary value or
choose a high-order interpolation. This separation lets a later numerical scheme
select first-order, limited second-order, or learned reconstruction while using
the same mesh connectivity.

## 6. Flux registers store integrated mismatch

Suppose the coarse level uses flux estimate `F_c` once over its time step, while a
fine level takes several substeps and uses `F_f`. `FluxRegister` accumulates

\[
I_c=\sum \Delta t_c A_f F_c,
\qquad
I_f=\sum_m \Delta t_{f,m} A_f F_{f,m}.
\]

Each register row corresponds to one coarse-fine face segment and stores the
oriented mismatch

\[
M=I_f-I_c.
\]

If `s_c` is the face-incidence sign viewed from the coarse cell, refluxing a cell
average applies

\[
Q_c \leftarrow Q_c - \frac{s_c M}{V_c}.
\]

For a cell integral, division by volume is omitted. Only the coarse cell is
corrected because the fine cell is assumed to have already used `I_f`. Multiple
fine segments on one coarse side naturally contribute multiple corrections to the
same coarse row.

The register can accumulate several fine substeps before `reflux()` is called.
It does not calculate either flux; a numerical or learned flux operator supplies
those values.

## 7. Topology transfer already handles mesh changes

`remap_state()` and `adapt_state_topology()` remain the transfer operators:

- parent to children: first-order prolongation;
- children to parent: volume-weighted restriction;
- optional cut-cell fluid measures;
- a conservation check before returning the new state.

The new hierarchy and face topology are rebuilt from the returned `TreeLayout`.
They are derived connectivity, so checkpoints do not need to store them twice.

## 8. Run the complete tiny example

```bash
uv run python examples/amr_finite_volume.py
```

It performs no CFD simulation. It checks two deliberately simple invariants:

1. a constant physical flux has zero discrete divergence across a mixed-level
   quadtree;
2. a one-unit coarse/fine flux mismatch is corrected so the composite conserved
   total returns from `-1` to `0`.

## 9. Deliberate limitations

- Face construction still compares leaf pairs and is quadratic.
- Faces currently cover Cartesian cell interfaces and outer domain boundaries;
  embedded obstacle fragments remain separate geometry objects.
- Ghost values and physical boundary conditions are not evaluated yet.
- No reconstruction, Riemann solver, diffusion operator, or time integrator is
  implemented.
- There is no patch/box storage, level scheduler, or automatic subcycling.
- Refluxing assumes the caller supplies consistently oriented coarse and fine
  flux estimates.

These boundaries are useful: the representation and conservation bookkeeping can
now be tested independently before a PDE solver is added.
