# Cut-cell and embedded-boundary quality

This note records the quality questions raised by the first circle and NACA 0012
geometry previews. It is a design reference for later work; none of the metrics
below is an acceptance criterion in the current package.

## 1. There are two different meshes to judge

Our background quadtree cells are axis-aligned squares. They cannot become
inverted, skewed, or non-orthogonal. Clipping one of those squares with an
obstacle, however, creates a **fluid control volume** that may be a tiny or
awkward polygon.

This gives us two separate quality questions:

1. Is the Cartesian hierarchy valid and graded well enough for neighbour
   stencils?
2. Is the clipped fluid region suitable for a stable and accurate numerical
   update?

The second question is the important one for an embedded-boundary finite-volume
solver. A perfectly regular background square can contain a very poor fluid
sliver.

The current example geometries already expose this issue. At commit
`9500e9e4`, the example policy produced the following diagnostic counts:

| Geometry | Cut cells | Minimum fluid fraction | `chi < 0.01` | `chi < 0.1` |
| --- | ---: | ---: | ---: | ---: |
| Circle | 76 | `7.47e-5` | 8 | 12 |
| NACA 0012 | 36 | `7.73e-3` | 2 | 10 |

These numbers do not mean the geometry is wrong. They mean that a future
explicit solver cannot treat every cut cell exactly like a full Cartesian cell.

## 2. Core embedded-boundary metrics

No single scalar describes cut-cell quality. We should compute a small vector of
metrics and let the numerical method decide what action each problem requires.

### Fluid volume fraction

In 2D, use area; in 3D, use volume:

```text
chi_i = fluid_measure_i / background_cell_measure_i
```

`chi` is the first small-cell warning. A conservative explicit update divides a
flux imbalance by the fluid volume, so a very small `chi` can impose a very small
stable time step.

For early visual diagnostics only, `chi < 0.1` can be marked as a warning and
`chi < 0.01` as critical. They are deliberately not solver acceptance limits.
Published merging and stabilization methods use problem-dependent thresholds,
often anywhere from a few percent to roughly one half of a full cell.

### Face aperture

For every Cartesian face, store the fraction open to fluid:

```text
alpha_f = open_face_measure / full_face_measure
```

The measure is an edge length in 2D and a face area in 3D. A small aperture can
produce a weak or ill-conditioned connection even when `chi` is not unusually
small. Volume fractions alone therefore do not describe the usable flux stencil.

### Boundary closure

A closed polygon or polyhedron should have zero net outward area vector. A
scale-free diagnostic is

```text
closure_error = norm(sum_f S_f) / sum_f norm(S_f)
```

where `S_f` is outward edge-normal times edge length in 2D, or outward face-area
vector in 3D. The sum must include both Cartesian apertures and the embedded
boundary. A large closure error usually indicates an intersection, orientation,
or bookkeeping bug rather than a cell that merely needs refinement.

### Centroid displacement

Store the true fluid centroid, not just the centre of the background cell, and
measure

```text
centroid_offset = distance(fluid_centroid, background_center) / cell_width
```

Large offsets make centre-to-centre interpolation less representative and can
increase skewness in a finite-volume stencil.

### Connectivity and components

A cut operation can create more than one disconnected fluid fragment inside one
background cell. Record the number of connected components and whether the cell
is single- or multi-valued. Our first solver should reject or refine multi-valued
cells rather than pretending that disconnected fragments share one state value.

### Compactness

Compactness is a useful secondary indication of a long, thin sliver:

```text
2D:  4 pi A / P^2
3D:  36 pi V^2 / S^3
```

Both expressions are one for a circle or sphere and decrease for elongated or
irregular shapes. They must not be treated as universal pass/fail metrics: even a
normal square has 2D compactness `pi/4`, and a physically valid boundary can
naturally create a low-compactness cut cell.

## 3. Geometry-approximation metrics

Cut-cell stability and surface accuracy are related but different. We should also
track how well the supplied polyline or triangle surface represents the intended
smooth geometry:

- maximum boundary distance or chord error;
- angle between discrete and reference normals;
- `curvature * cell_width`, a dimensionless resolution measure;
- number and length/area of embedded-boundary fragments in a cell;
- tangential spacing near a sharp feature or thin gap.

High curvature or multiple crossings can justify refinement. In contrast, merely
refining a tiny sliver does **not** guarantee a larger `chi`: the boundary may cut
one of the children even closer to its corner. Refine to resolve geometry;
stabilize, merge, or redistribute to handle small-cell time-step restrictions.

## 4. Stencil quality once physical faces exist

The following metrics become meaningful after we build one numerical face per
fluid connection:

- non-orthogonality between the face normal and the line joining cell centroids;
- face-centre skewness relative to the centroid-to-centroid interpolation line;
- interpolation weight and centre-to-face distance ratio;
- condition number of a local least-squares gradient reconstruction;
- neighbour level/size ratio and satisfaction of 2:1 balance;
- number and directional spread of usable neighbours.

OpenFOAM's familiar non-orthogonality, skewness, concavity, determinant, and face
weight checks are valuable here. They are not direct measures of our untouched
background squares; they apply to the effective control volumes and flux
geometry.

For arbitrary cut polygons and polyhedra, least-squares conditioning is often
more informative than a triangle or tetrahedron aspect ratio because the solver
uses a reconstruction stencil, not a simplex element.

## 5. Additional checks needed in 3D

Before clipping, an input triangle surface should be checked for:

- watertightness and consistent orientation;
- two-manifold edges;
- self-intersections;
- duplicate or degenerate triangles;
- triangles with poor scaled Jacobian or very small angles.

After clipping, each fluid polyhedron should have closed intersection loops,
positive signed volume, consistently oriented faces, and valid face
triangulations. We should initially reject cells containing disconnected fluid
components or non-manifold boundaries.

Body-fitted element measures such as signed/scaled Jacobian, dihedral angles,
warpage, concavity, aspect ratio, and radius ratio remain useful for imported
triangle, tetrahedron, or hexahedron meshes. They are supplementary—not a
replacement for `chi`, apertures, closure, and stencil conditioning in a cut-cell
method.

## 6. A quality metric should select an action

| Detected problem | First response |
| --- | --- |
| Invalid topology, negative measure, or poor closure | Reject as a geometry bug |
| Under-resolved curvature, thin gap, or multiple crossings | Refine geometry |
| Very small `chi` or face aperture | Merge/agglomerate or use conservative redistribution |
| Ill-conditioned reconstruction | Enlarge/regularize the stencil or merge cells |
| High non-orthogonality or skewness | Use a corrected flux/reconstruction or merge |
| Large level jump | Apply 2:1 balancing |

This separation is important. A generic rule such as "refine every bad cell"
can increase cell count without fixing the actual numerical problem.

## 7. Deferred implementation slice

When we return to cut quality, the smallest useful addition is:

1. add fluid centroid, true embedded-boundary centroid and measure, per-face
   apertures, and connected-component count to snapped geometry;
2. add a `CutCellQuality2D` value object and an aggregate report;
3. add visualization modes for `chi`, minimum aperture, closure error, and
   centroid offset;
4. add translation tests in which the same circle moves by subcell increments;
5. add difficult corner, tangent, thin-gap, and double-crossing cases;
6. keep warning thresholds in a policy object rather than in geometry code.

Only after those diagnostics should we choose a small-cell treatment for the
first transport solver.

## Sources

- [AMReX embedded-boundary data and small-cell discussion](https://amrex-codes.github.io/amrex/docs_html/EB.html)
- [OpenFOAM mesh description and validity](https://www.openfoam.com/documentation/user-guide/4-mesh-generation-and-conversion/4.1-mesh-description)
- [OpenFOAM mesh-quality controls](https://doc.openfoam.com/2606/tools/pre-processing/mesh/generation/snappyhexmesh/meshquality/)
- [OpenFOAM `surfaceCheck`](https://www.openfoam.com/documentation/guides/latest/man/surfaceCheck.html)
- [Gmsh mesh-quality measures](https://gmsh.info/doc/texinfo/gmsh.html)
- [Sandia Cubit triangle metrics](https://cubit.sandia.gov/files/cubit/17.06/help_manual/WebHelp/mesh_generation/mesh_quality_assessment/triangular_metrics.htm)
- [Sandia Cubit tetrahedron metrics](https://www.sandia.gov/files/cubit/15.2/help_manual/WebHelp/mesh_generation/mesh_quality_assessment/tetrahedral_metrics.htm)
- [CGAL polygon-mesh processing and repair](https://doc.cgal.org/latest/Polygon_mesh_processing/index.html)
- [Cut-cell stabilization review/example](https://www.sciencedirect.com/science/article/pii/S0168927415000896)
- [State redistribution for cut cells](https://arxiv.org/abs/2005.05734)
