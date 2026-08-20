# Geometry-driven AMR and embedded-boundary snapping

This step answers a question that comes before solving any flow equation:

> Given an obstacle, where should the Cartesian tree be fine, where may it be
> coarse, and how is the obstacle represented inside a cell?

The implementation is deliberately two-dimensional and small. It gives us a
reference algorithm that is easy to inspect before we optimize it or ask a learned
model to participate.

## 1. The geometry authority is a closed polyline

`Obstacle2D` stores a valid closed polygon. A circle is represented by a regular
polygon and a NACA four-digit airfoil is sampled into upper and lower polylines.
This is similar in spirit to an STL workflow: the mesher is exact with respect to
the supplied segments, while agreement with the original smooth curve depends on
how densely that curve was sampled.

For each boundary vertex we calculate three local characteristics:

- the change in tangent direction, or **turning angle**, `delta_theta`;
- the average length `s` of the two adjacent segments;
- the discrete curvature estimate `kappa = delta_theta / s`.

A regular polygon approximating a circle of radius `r` therefore has
`kappa` close to `1/r`. A NACA airfoil has much higher curvature at its nose, and
its sharp trailing edge is detected as a large change in direction.

The NACA constructor uses cosine-spaced points. Cosine spacing deliberately puts
more geometry samples near the leading and trailing edges, where the shape changes
most rapidly.

## 2. What "snapping" means in this first implementation

OpenFOAM's `snappyHexMesh` first refines a background mesh and then morphs mesh
vertices toward a surface. Moving vertices requires smoothing, mesh-quality tests,
and rollback when a proposed move creates a poor cell.

Our first version uses a simpler **embedded-boundary** interpretation:

1. The quadtree rectangles never move.
2. An obstacle polyline is intersected with every leaf rectangle.
3. A cut cell stores the exact polyline fragment and intersection coordinates.
4. The cell's usable fluid region is the rectangle minus the solid obstacle.

The intersection coordinates are the snapped points: they lie on the input
polyline. This preserves the tree hierarchy and cannot create an inverted cell.
It is not yet a body-fitted quadrilateral mesh.

For cell `i`, the fluid fraction is

```text
chi_i = area(cell_i minus obstacle) / area(cell_i).
```

The classification is then:

```text
chi_i = 1       fluid cell
0 < chi_i < 1   cut cell
chi_i = 0       solid cell
```

Because the area is obtained by geometric clipping, summing
`chi_i * area(cell_i)` recovers the total fluid area up to floating-point roundoff.

Each leaf also stores the signed distance from its centre, the closest boundary
point, and the outward obstacle normal. Those values will later support boundary
conditions and geometry-aware graph features.

## 3. Common geometry-based refinement rules

Production meshers usually combine several rules instead of relying on one
indicator. We currently take the maximum requested level from the following rules.

### Surface intersection

Any cell crossed by the boundary receives at least `boundary_level`. Refining
intersected cells is the basic castellation step.

### Distance bands

Distance bands request progressively coarser levels away from the wall. For
example:

```python
distance_bands=(
    DistanceBand(0.04, 5),
    DistanceBand(0.12, 4),
    DistanceBand(0.30, 3),
)
```

This means "level 5 within 0.04 length units, level 4 within 0.12, and level 3
within 0.30." It is analogous to OpenFOAM refinement regions/features and Gmsh
`Distance` plus `Threshold` size fields.

### Curvature

High-curvature regions need shorter cells to approximate a smooth curve. For a
short circular arc, the distance between an arc and its chord (the **sagitta**) is
approximately

```text
error ~= kappa * h^2 / 8.
```

Given an allowed `max_chord_error`, we invert that relation:

```text
h_target <= sqrt(8 * max_chord_error / kappa).
```

The tree chooses the first level whose physical cell width is no larger than
`h_target`. This is a geometric error estimate, not yet a CFD discretization-error
estimate.

### Sharp features

If the turning angle exceeds `feature_angle_degrees`, cells close to that vertex
receive `feature_level`. This catches corners and an airfoil trailing edge. The
criterion is local: a flat part of the same obstacle can stay at the ordinary
boundary level.

### Grading and 2:1 balance

After explicit tagging, `AdaptiveTree.balance()` ensures face neighbours differ by
at most one level. This supplies a one-cell transition between very different
resolutions. Later finite-volume interpolation and flux conservation are much
simpler on a 2:1-balanced tree.

Narrow-gap or local-feature-size detection is another common geometry rule. It is
not implemented yet because our first circle and single-airfoil cases contain no
opposing nearby surfaces. It should be added before multi-obstacle cases.

## 4. Refinement algorithm

For every current leaf:

1. Build its physical rectangle.
2. Evaluate intersection, distance, local curvature, and sharp-feature rules.
3. Let `target_level` be the largest level requested by any rule.
4. If `cell.level < target_level`, split it into four children.
5. Repeat because the new children see more local geometry than their parent did.
6. Enforce 2:1 balance.

This is a direct `O(number of cells * number of boundary segments)` reference
implementation. A spatial index will be needed for large geometries, but optimizing
before we have trustworthy tests would hide the core algorithm.

## 5. Coarsening algorithm

Quadtree coarsening must preserve the hierarchy. We may replace four children by
their parent only when:

1. all four siblings are leaves;
2. the parent itself satisfies the local geometry target;
3. the result still satisfies 2:1 balance.

The second condition is important. It lets solid interiors and far-field fluid
become coarse, but prevents a parent that crosses the obstacle from erasing needed
surface resolution.

For moving geometry, a boundary can hover around a threshold and cause repeated
refine/coarsen cycles. `coarsening_hysteresis` expands distance and feature bands
only during the coarsening test. A cell must move clearly outside the refinement
zone before it is removed.

## 6. Classic first geometries

- **Circle/cylinder:** constant analytical curvature, symmetry, known area, and
  simple normals make it the best first verification geometry.
- **NACA 0012:** adds a highly curved leading edge, thin solid region, and sharp
  trailing edge without requiring an external geometry file.
- **Lid-driven cavity:** excellent for the first incompressible solver, but it has
  no internal obstacle and therefore does not exercise snapping.
- **Complex animal/CAD surface:** useful later for importer and spatial-index
  stress tests. The remembered OpenFOAM dolphin is most likely a user-supplied
  Harpoon mesh from a 2005 conversion discussion, not a bundled tutorial. Its
  original download and license could not be recovered, so it should not be copied
  into this repository. The evidence and a licensed modern alternative are recorded
  in [`docs/research/01-openfoam-dolphin.md`](../research/01-openfoam-dolphin.md).

## 7. Run the tiny visual check

These commands perform geometry operations only; they do not run OpenFOAM or a CFD
solver:

```bash
uv run python examples/geometry_amr.py --shape circle --output circle.svg
uv run python examples/geometry_amr.py --shape naca0012 --output naca0012.svg
```

White cells are fluid, grey cells are solid, and orange cells are cut. The black
line is the input obstacle polyline. Add `--show-points --show-normals` to inspect
where cell edges meet the obstacle and which outward normal is stored for each cut
cell. See [`03-visualizing-geometry.md`](03-visualizing-geometry.md) for the layers
and color modes.

## 8. Numerical issue deliberately exposed by cut cells

An obstacle can leave a very small fluid sliver in a Cartesian cell. In an explicit
finite-volume method, its small volume can impose a severe stable time-step limit.
Before advancing a CFD solution we will need a documented policy such as cell
merging, conservative flux redistribution, or a stabilized cut-cell update.

That is exactly why geometry metadata belongs in the numerical framework rather
than being hidden inside a preprocessing script.

## References used for this design

- [OpenFOAM snappyHexMesh overview](https://doc.openfoam.com/2312/tools/pre-processing/mesh/generation/snappyhexmesh/)
- [OpenFOAM castellation controls](https://doc.openfoam.com/2606/tools/pre-processing/mesh/generation/snappyhexmesh/castellation/)
- [OpenFOAM snapping controls](https://doc.openfoam.com/2306/tools/pre-processing/mesh/generation/snappyhexmesh/snapping/)
- [Gmsh mesh-size fields and curvature sizing](https://gmsh.info/doc/texinfo/)
- [AMReX grid creation and tag buffers](https://amrex-codes.github.io/amrex/docs_html/GridCreation.html)
- [AMReX embedded-boundary geometry](https://amrex-codes.github.io/amrex/docs_html/EB.html)
- [NASA OpenVSP NACA four-series reference](https://www.nasa.gov/reference/openvsp-cross-sections/)
