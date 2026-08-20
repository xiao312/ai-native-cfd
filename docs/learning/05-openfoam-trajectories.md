# Learning note 05: from an OpenFOAM trajectory to our quadtree

This experiment reverses the usual development order. Instead of implementing a
new evolution algorithm first, OpenFOAM supplies a trusted sequence of states and
our Python package learns how to represent that sequence exactly.

## 1. The reference problem

The lid-driven cavity is a square, two-dimensional domain. Three walls are fixed
and the top wall moves from left to right. Viscosity transmits that motion into the
fluid and creates a circulating vortex.

We use the OpenFOAM Foundation v7 `icoFoam` tutorial because it is transient,
laminar, small, and based on one Cartesian block. OpenFOAM still stores a 2D case
as a one-cell-thick 3D mesh; the front and back patches use the special `empty`
boundary condition.

The Reynolds number is deliberately small:

\[
Re = \frac{U L}{\nu} = \frac{1\times0.1}{0.01}=10.
\]

That gives a smooth laminar flow, so our first data-path experiment is not mixed
up with turbulence modelling, shocks, or complex geometry.

## 2. Why 64 by 64?

Our quadtree divides each coordinate direction by two at every level. Level 6 has

\[
2^6 \times 2^6 = 64 \times 64 = 4096
\]

leaves. Choosing the same OpenFOAM resolution makes every source cell coincide
with exactly one quadtree leaf. The first import therefore tests ordering rather
than interpolation.

The physical cavity width is `0.1 m`, so a cell width is

\[
h = \frac{0.1}{64} = 0.0015625\;\mathrm{m}.
\]

With a lid speed of `1 m/s` and `dt = 0.00125 s`, the worst simple Courant
estimate is

\[
Co \approx \frac{|U|\,dt}{h}=0.8.
\]

The original tutorial time step of `0.005 s` would give approximately `Co = 3.2`
on this finer grid, so it cannot simply be retained.

## 3. What OpenFOAM records

Setting `writeInterval 1` creates a directory after every physical time step. The
run advances from `0` to `0.5 s` in steps of `0.00125 s`, giving 400 updates and
401 stored states including the initial condition.

Each time directory contains:

- `U`, the three-component OpenFOAM velocity field;
- `p`, kinematic pressure with dimensions of `m^2/s^2`.

These are physical time states, not the intermediate pressure-corrector or linear
solver iterations inside each `icoFoam` step.

The importer currently stores each field's `internalField`: one value per volume
cell. Boundary-patch values remain in the raw OpenFOAM files. This is why the
initial cell-centred velocity is zero even though the top-wall boundary condition
already prescribes a lid speed of `1 m/s`; diffusion carries that motion into the
cell interiors during subsequent steps.

## 4. Mapping source rows to tree rows

OpenFOAM's internal cell ordering is its own implementation detail. We therefore
ask OpenFOAM to write the cell-centre field `C` once. For source centre `(x, y)`,
the logical quadtree indices are

\[
i=\left\lfloor64\frac{x-x_0}{L_x}\right\rfloor,
\qquad
j=\left\lfloor64\frac{y-y_0}{L_y}\right\rfloor.
\]

The importer verifies that the source centre agrees with the expected centre of
cell `(level=6, index=(i, j))`. It then uses `TreeLayout.index` to reorder the
source data into our deterministic Morton-prefix order.

No learned model, interpolation, or native OpenFOAM C++ object is involved.

## 5. Fixed trajectory storage

All 401 states share one mesh, so topology is stored once:

```text
trajectory/
  metadata.json
  topology.npz
  trajectory.npz
```

The main arrays have shapes:

```text
times: (401,)
steps: (401,)
U:     (401, 4096, 2)
p:     (401, 4096)
```

The out-of-plane component of `U` is checked before it is dropped. Raw pressure is
preserved. For visualization or learning, a volume-weighted pressure mean can be
removed because incompressible pressure has an arbitrary reference offset.

The two top corners join a moving lid to stationary side walls, so velocity is
discontinuous there. The resulting corner pressure peaks dominate a full-range
color map. Pressure previews therefore use a symmetric color limit based on the
99th percentile of absolute centered pressure. Values outside that range are
color-clipped only in the SVG; stored trajectory values are never clipped.

## 6. Running and converting

On the Wuzhen SCNet account, submit the one-core job from the repository root:

```bash
sbatch scripts/scnet-wuzhen/run_cavity64.slurm
```

There is also a Kunshan wrapper at
`scripts/agent-scnet/run_cavity64.slurm`. Cluster wrappers contain only resource
and software-environment details; both call one shared numerical case runner.

After copying the completed case back to a local ignored data directory, convert
it with:

```bash
uv run python examples/import_cavity_trajectory.py \
  .data/openfoam/cavity64 \
  .data/processed/cavity64 \
  --previews artifacts/openfoam/cavity64
```

The raw OpenFOAM time directories and compact numerical trajectory are generated
data and are not committed. Small SVG previews may be retained as human-readable
evidence.

## 7. What comes next

The first converted trajectory remains a uniform level-6 tree. The next experiment
will compute one refinement envelope over the entire time series, keep fine cells
near the moving lid and large velocity changes, and conservatively coarsen smooth
regions. Keeping one shared adaptive topology initially isolates AMR projection
from the harder problem of changing the graph at every time step.
