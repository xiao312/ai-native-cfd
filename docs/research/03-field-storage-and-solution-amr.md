# Physical fields and solution-driven AMR

This note answers the next two design questions for `ai-native-cfd`:

1. How should physical values be represented on an adaptive tree?
2. How should those values request refinement and coarsening?

The recommendation is intentionally smaller than AMReX, OpenFOAM, or Basilisk.
We borrow their durable ideas without reproducing their native data structures.

> **Implementation status:** the first two slices described below are implemented
> with cell-centred NumPy fields, constant conservative prolongation,
> volume-weighted restriction, safe checkpoints, value-range and neighbour-jump
> indicators, and a first-order Haar-like wavelet detail. See the fourth
> [learning note](../learning/04-physical-fields-and-solution-amr.md) for the
> concrete APIs and current limitations.

## 1. The central separation

Keep four concepts distinct:

```text
tree topology -> leaf layout -> physical fields -> adaptation policy
```

- **Tree topology** says which cells exist and how they are related.
- **Leaf layout** assigns each current leaf to one array row.
- **Physical fields** store values on that particular layout.
- **Adaptation policy** reads fields, produces tags, and asks for a new layout.

Do not add `pressure`, `velocity`, or a general dictionary of values to `Cell`.
`Cell` is a small immutable spatial identity. Millions of Python objects with
independent field dictionaries would consume excessive memory, prevent efficient
NumPy operations, and make it difficult to detect stale data after adaptation.

AMReX follows the same broad separation at a larger scale: a `BoxArray` describes
the regions on one level, while `MultiFab` owns component arrays and ghost cells
on that layout. p4est similarly manages the forest, refinement, balancing, and
partitioning while allowing application data to be associated with leaves.

## 2. Small in-memory model for this repository

The first implementation should use flat NumPy arrays over a deterministic leaf
snapshot. A possible vocabulary is:

```text
TreeLayout
    cells: tuple[Cell, ...]
    cell_to_index: dict[Cell, int]
    topology_id: immutable fingerprint/revision

FieldSpec
    name
    component_names
    location
    value_representation
    dtype
    unit metadata

Field
    spec
    layout topology_id
    values: numpy.ndarray

State
    tree/layout
    time and step
    mapping from field name to Field
```

For a scalar, `values.shape == (number_of_leaves,)`. For a vector, the first
version can use `(number_of_leaves, number_of_components)`. The exact memory order
can be benchmarked later; the important first decision is one typed array per
field rather than one Python value per cell.

Array position is not cell identity. Code should resolve a `Cell` through
`cell_to_index`, and files must store the ordered cell list alongside the arrays.
After refinement, a cell can move to a different row even if it still exists.

The existing snapped-geometry renderer already rejects a snapshot whose cells no
longer match the tree. Fields should enforce the same rule explicitly through a
topology identifier, not merely by comparing array length.

## 3. Field location must be explicit

CFD variables do not all live in the same place. Reserve an enum such as:

```text
CELL
FACE_X, FACE_Y, FACE_Z
NODE
EMBEDDED_BOUNDARY
```

Start with **cell-centred scalar fields only**. This is sufficient for a
manufactured scalar-transport case and keeps the first transfer rules clear.
Later:

- conserved state and pressure can be cell-centred;
- normal fluxes and a staggered velocity can be face-centred;
- geometric boundary conditions can use embedded-boundary values;
- a finite-element experiment may need node-centred degrees of freedom.

AMReX gives cell state and directional face fluxes different index types, and
Basilisk distinguishes centred, face, and vertex fields because interpolation and
boundary operations depend on the staggering. Encoding location as metadata now
prevents an accidental cell/face mix-up later.

## 4. Store what a finite-volume value means

For the first solver, a cell value should mean a **fluid-volume average**:

```text
q_i = (1 / V_i) integral_over_fluid_cell(q dV)
V_i = chi_i * background_cell_volume
```

It is not merely a point sample at the background-cell centre. The corresponding
extensive amount is

```text
Q = sum_i(q_i * V_i).
```

This distinction controls every refinement transfer and conservation test. A
future `value_representation` should distinguish at least:

- `CELL_AVERAGE`: restrict using volume weights;
- `CELL_INTEGRAL`: restrict by summing;
- `POINT_VALUE`: interpolate as a sample, with no implied conservation law.

The word "scalar" only describes the number of components; it does not say which
of these meanings applies.

Field metadata should also include component names, numeric dtype, and physical
units or SI dimension exponents. openPMD is a useful future interoperability
reference because it records axes, grid spacing/offset, relative position within
an element, time, and SI unit dimensions. We do not need to adopt its storage API
for this small prototype.

## 5. Conservative transfer when the tree changes

### Coarsening: restriction

For child cell averages, the parent value is the fluid-volume-weighted mean:

```text
q_parent = sum_children(V_child * q_child) / sum_children(V_child)
```

For cell integrals, simply sum the child values. Solid children have zero fluid
volume and do not contribute. The clipped child volumes must sum to the clipped
parent volume within geometric tolerance.

### Refinement: prolongation

The safest first-order operation copies the parent average to every fluid child:

```text
q_child = q_parent
```

Although inaccurate for a varying field, this is conservative because the child
fluid volumes partition the parent fluid volume. It gives us a simple oracle for
the data model.

A later second-order operation can reconstruct

```text
q(x) = q_parent + limited_gradient dot (x - parent_centroid)
```

and average that reconstruction over each child. A final weighted correction
should force the child integral to equal the parent integral exactly. The gradient
needs a limiter so prolongation does not create new extrema near shocks or bounded
fractions.

Basilisk makes restriction and prolongation operations attributes of a field; it
uses specialised versions for volume fractions, embedded boundaries, and
divergence-free face velocity. That is a good long-term direction: transfer is a
property of a field's numerical meaning, not one universal tree operation.

### Flux conservation comes later

Conservative state transfer is necessary but not sufficient for time-dependent
finite-volume AMR. At a coarse-fine interface, one coarse face covers several fine
faces. After advancing, **refluxing** corrects the neighbouring coarse cell by the
difference between its coarse flux and the sum of fine fluxes. Berger and Colella
introduced this mechanism to preserve conservation across levels.

We should add refluxing only after physical faces and a scalar finite-volume
update exist. It does not belong in the first field container.

## 6. Adaptation should be a transaction

Mutating the leaf set before transferring fields can silently associate a value
with the wrong cell. Use this sequence instead:

1. freeze the old tree layout and state;
2. fill any neighbour/ghost data needed by indicators;
3. compute an indicator score for every old leaf;
4. turn scores into refine, keep, and coarsen requests;
5. apply geometry floors, limits, buffers, sibling rules, and 2:1 balance;
6. build the new layout;
7. copy unchanged values and restrict/prolong changed families;
8. verify shapes, finiteness, bounds, and conserved totals;
9. publish the new state only if every check succeeds.

This also gives us a clean future hook for learned refinement: a learned model can
propose scores without owning topology mutation or field transfer.

## 7. Common solution-based indicators

An **indicator** is a local score suggesting where resolution is useful. It is not
the adaptation policy itself.

### Value or interval threshold

Examples are `q > threshold` or `lower < q < upper`. OpenFOAM's
`dynamicRefineFvMesh`, for example, selects candidates from a configured field and
lower/upper value range. This is effective for a known interface fraction or a
specific physical regime, but it is not a general error estimate.

Use it first as a debugging indicator because its expected tags are obvious.

### Neighbour jump or gradient

A simple dimensionless score is

```text
eta_i = max_neighbours(abs(q_j - q_i)) / q_scale
```

or, equivalently, a physical-gradient estimate multiplied by local cell width.
It detects fronts, shear layers, and shocks. Clawpack's default flagger uses
undivided differences to neighbouring values. On a nonuniform tree, differences
must use physical centre distances; when several fine cells share a coarse face,
shared-face length/area should weight their contribution.

This should be our first physically useful indicator after face-neighbour access
and field storage exist.

### Normalized second derivative (Lohner-type)

A gradient indicator also refines a perfectly linear field even though a linear
reconstruction can represent it exactly. A curvature/second-difference indicator
focuses on changes in the gradient. In one dimension, the common form is

```text
eta_i = abs(q[i+1] - 2*q[i] + q[i-1])
        / (abs(q[i+1] - q[i]) + abs(q[i] - q[i-1])
           + epsilon * local_magnitude)
```

The normalization makes it dimensionless and the small filter prevents tiny
numerical ripples around zero from requesting refinement. FLASH/PARAMESH uses a
multidimensional version and allows several chosen fields.

### Wavelet or prediction defect

Tree grids offer a particularly natural estimate:

1. restrict fine values to a coarser representation;
2. prolong that coarse representation back to the fine cells;
3. measure the detail that was lost.

```text
eta_i = abs(q_i - prolong(restrict(q))_i) / q_scale
```

Basilisk's `adapt_wavelet()` refines and coarsens selected fields against
user-provided error tolerances. This measures representation error rather than
only field magnitude and should be our first more sophisticated indicator after
restriction/prolongation are trustworthy.

### Richardson or truncation-error estimate

Compare approximations made at two resolutions (or with different step sizes) and
use the difference to estimate error. AMRClaw supports Richardson flagging. It is
closer to the actual discretization error but costs extra numerical work and
requires a solver, so it is not an initial implementation target.

### Physics-specific sensors

Production CFD often refines combinations such as:

- density or pressure jumps for compressible shocks;
- vorticity or velocity-gradient magnitude for shear layers and wakes;
- volume fraction or signed distance for interfaces;
- temperature, heat release, reaction rate, or species gradients for flames;
- wall distance and estimated `y+` for wall treatment.

These are useful features, not universal error estimates. Each needs a documented
scale and physical purpose.

### Goal-oriented or adjoint indicators

If the quantity of interest is drag, lift, or a downstream probe, the largest
local solution error is not necessarily the most important error. An adjoint can
weight local errors by their influence on that output. Clawpack includes adjoint
flagging, and SU2 supports goal-oriented adaptation. This is powerful but belongs
well after the baseline solvers.

## 8. Policy around the indicator

A robust policy adds constraints that a raw score does not know about:

- combine fields after nondimensionalizing each by its tolerance;
- allow **any** important field to request refinement;
- allow coarsening only when **all** fields and all siblings are safely below
  their coarsening thresholds;
- use hysteresis, with `coarsen_threshold < refine_threshold`, to prevent mesh
  chatter;
- grow/buffer tags so moving features do not leave the refined zone before the
  next regrid;
- enforce minimum/maximum levels and geometry-required levels;
- enforce 2:1 balance after requested changes;
- respect a deterministic maximum-cell budget and prioritize the largest
  normalized benefit when necessary;
- optionally require a cell to remain unchanged for several adaptation cycles
  before coarsening.

AMReX explicitly grows error tags and enforces proper nesting. AMRClaw recommends
choosing its buffer width with the regrid interval so waves cannot outrun the fine
region. OpenFOAM exposes buffer layers, a maximum refinement level, and a maximum
cell count. These are policy controls, not new error indicators.

For geometry plus physics, the requested level should be the maximum:

```text
target_level(cell) = max(
    geometry_required_level(cell),
    solution_requested_level(cell),
    user_forced_level(cell),
)
```

A physics-driven coarsener must never erase the minimum resolution needed to
represent the obstacle.

## 9. Ghost and neighbour values

Gradient and curvature indicators need values beyond one cell. Large AMR codes
usually allocate ghost/guard cells, then fill them from same-level neighbours,
coarse-to-fine interpolation, periodic copies, and physical boundary conditions.
AMReX's valid-versus-ghost distinction and `FillPatch` operations are a useful
model.

Our tiny serial tree does not need stored ghost arrays yet. It can query actual
leaf neighbours directly and apply boundary rules explicitly. Ghost storage
should be added when block arrays, domain decomposition, or stencil performance
requires it; ghost values are derived cache data, never authoritative solution
state.

## 10. Disk storage and interchange

Keep restart storage separate from visualization export.

The smallest transparent checkpoint can be a directory containing:

```text
metadata.json       schema version, dimension, origin, extent, time, step,
                    field specifications, geometry reference/fingerprint
topology.npz        leaf levels and integer indices in exact row order
fields.npz          one named numeric array per field
```

Do not use pickle for a long-lived scientific format. On load, reconstruct the
tree/layout first, allocate fields against it, then read values and validate every
shape and topology identifier. This mirrors AMReX's level-by-level checkpoint
order: restore the grid description, construct data containers, then load their
data.

Plot files may omit restart-only details and may convert staggered fields for a
visualization tool. Later we can add VTK/OpenFOAM export and evaluate HDF5, Zarr,
or openPMD-style metadata when arrays no longer fit comfortably in one process.
The in-memory API should not depend on the first file format.

## 11. Recommended implementation order

### Slice A: field foundations

1. `TreeLayout` with deterministic leaves and a topology fingerprint.
2. `FieldLocation.CELL` and `FieldSpec` for a scalar cell average.
3. `CellField` backed by a one-dimensional NumPy array.
4. `State` with time, step, and a field registry.
5. constant first-order prolongation and volume-weighted restriction.
6. conservative refine/coarsen transaction and stale-layout checks.
7. a small checkpoint round trip.

### Slice B: deterministic solution AMR

1. an `Indicator` protocol returning one score per leaf;
2. a value-range indicator as the simplest debugging implementation;
3. a normalized neighbour-jump indicator;
4. independent `AdaptationPolicy` thresholds, hysteresis, limits, and buffers;
5. combination with the existing geometry-required level and 2:1 balance;
6. a Gaussian or sinusoidal manufactured scalar field for deterministic tests;
7. a wavelet-detail indicator after transfer tests are solid.

### Essential tests

- a field rejects the wrong shape, location, or stale topology;
- constant fields survive arbitrary refine/coarsen cycles exactly;
- volume-weighted totals are conserved, including cut-cell volumes;
- restriction and prolongation are deterministic;
- a constant field produces zero jump/wavelet score;
- tags appear around a known manufactured feature and nowhere else;
- hysteresis prevents immediate refine/coarsen oscillation;
- cell budgets, maximum levels, geometry floors, and 2:1 balance always hold;
- checkpoint save/load reproduces topology, metadata, and values.

This order gives us useful physical data on the hierarchy without committing to a
PDE solver or an OpenFOAM-native representation.

## Primary sources

- [AMReX `MultiFab`, field centering, and ghost-cell data](https://amrex-codes.github.io/amrex/docs_html/Basics.html)
- [AMReX tagging, level creation, and `FillPatch`](https://amrex-codes.github.io/amrex/docs_html/AmrCore.html)
- [AMReX grid creation, buffers, and clustering](https://amrex-codes.github.io/amrex/docs_html/GridCreation.html)
- [AMReX plotfile and checkpoint organization](https://amrex-codes.github.io/amrex/docs_html/IO.html)
- [Basilisk centred, face, and vertex fields](https://basilisk.fr/Basilisk%20C)
- [Basilisk restriction and prolongation operations](https://basilisk.fr/src/grid/multigrid-common.h)
- [Basilisk adaptive-wavelet algorithm](https://basilisk.fr/sandbox/Antoonvh/the_adaptive_wavelet_algorithm)
- [Basilisk transfer operators for embedded boundaries](https://basilisk.fr/src/embed-tree.h)
- [p4est architecture and typical refine/coarsen/balance workflow](https://github.com/cburstedde/p4est)
- [p4est scalable forest-of-octrees algorithms](https://epubs.siam.org/doi/10.1137/100791634)
- [OpenFOAM `dynamicRefineFvMesh` configuration and source](https://api.openfoam.com/2606/classFoam_1_1dynamicRefineFvMesh.html)
- [FLASH/PARAMESH normalized second-derivative criterion](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node60.html)
- [AMRClaw gradient, Richardson, buffer, and region controls](https://www.clawpack.org/v5.10.x/setrun_amrclaw.html)
- [AMRClaw adjoint flagging](https://www.clawpack.org/v5.10.x/adjoint.html)
- [Berger--Colella conservative AMR and refluxing](https://crd.lbl.gov/assets/pubs_presos/AMCS/ANAG/A113.pdf)
- [openPMD mesh metadata and units](https://www.openpmd.org/openPMD-api/classopen_p_m_d_1_1_mesh.html)
- [openPMD field position within a mesh element](https://www.openpmd.org/openPMD-api/classopen_p_m_d_1_1_mesh_record_component.html)
