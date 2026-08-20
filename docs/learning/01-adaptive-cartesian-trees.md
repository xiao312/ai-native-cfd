# Adaptive Cartesian trees from first principles

This note explains the algorithms implemented in `aicfd.representation`. The code
is intentionally small enough to read alongside the explanation.

## 1. One rule works in every dimension

Start with one root cell covering the computational domain. Refining a cell halves
it along every coordinate axis.

| Dimension | Common name | Children after refinement |
| --- | --- | ---: |
| 1D | binary tree | 2 |
| 2D | quadtree | 4 |
| 3D | octree | 8 |

In `d` dimensions, the number of children is therefore `2**d`. A single
dimension-independent implementation is less error-prone than maintaining three
nearly identical versions.

## 2. Identifying a cell without floating-point coordinates

A `Cell` stores two pieces of information:

```text
(level, integer index)
```

At level `l`, each axis has `2**l` possible cell positions. In one normalized
dimension, integer index `i` represents

```text
[i / 2**l, (i + 1) / 2**l].
```

For example, the 2D cell

```text
Cell(level=2, index=(2, 1))
```

occupies `[0.5, 0.75] x [0.25, 0.5]` in the unit square. Integer indices make
parent and child operations exact:

```text
parent index = child index // 2
child index  = 2 * parent index + a zero-or-one offset
```

No tolerance is needed to determine family relationships.

## 3. The leaf set is the current mesh

A tree contains internal nodes and leaves:

- an internal node has been refined;
- a leaf has not been refined and is therefore a current computational cell.

The first implementation stores only the leaf set. Refining a leaf removes it and
adds all its children. Coarsening removes a complete sibling family and restores
its parent.

Internal nodes are recovered by repeatedly following every leaf's `parent`. This
costs more than maintaining several synchronized indices, but it gives us one
authoritative data structure while the algorithms are still being validated.

## 4. Morton codes

At each refinement level, every axis contributes one zero-or-one bit. Those bits
identify which child was selected. Appending the child numbers down the tree gives
a Morton code.

We use `(dimension, level, Morton code)` as a readable stable identity. The level
is essential: a Morton code alone does not distinguish a parent from a descendant
whose remaining path bits happen to be zero.

Morton ordering will later help us place spatially related cells near each other in
arrays and files. It does not replace explicit parent-child or face connectivity.

## 5. Face neighbours on a nonuniform mesh

Two cells are face neighbours when:

1. their intervals touch on exactly one axis; and
2. their intervals overlap with positive length on every other axis.

Touching on two axes in 2D means meeting only at a corner, so those cells are not
face neighbours. In 3D, touching on two or three axes means sharing only an edge or
corner.

Cells at different levels are compared on a common integer grid. If the finer cell
is at level `L`, a level-`l` coordinate is multiplied by `2**(L-l)`. This preserves
exact comparisons and avoids floating-point roundoff.

The current implementation compares every leaf pair. This is `O(N**2)` and will
become too slow for large meshes. It is nevertheless a useful correctness oracle.
Later implementations can use Morton ranges, per-level hash maps, or a compiled
backend and must reproduce the same answers.

## 6. Why 2:1 balance exists

A tree is 2:1 face-balanced when neighbouring cells differ by at most one
refinement level. Equivalently, their widths differ by at most a factor of two.

Without this restriction, one large face might touch many tiny cells. Numerical
stencils, interpolation, and flux accounting become increasingly complicated.

The simple balancing algorithm repeatedly:

1. finds neighbouring leaves whose levels differ by more than one;
2. marks the coarser leaf;
3. refines all marked leaves;
4. repeats until there are no violations.

Refining one cell can create another imbalance nearby, which is why one pass is not
always sufficient. The algorithm terminates because it only refines toward levels
that already exist in the local mesh.

## 7. Invariants we test

After any legal sequence of operations:

- every leaf has the tree's dimension;
- no leaf is an ancestor of another leaf;
- leaves cover the root exactly;
- refinement creates exactly `2**dimension` children;
- coarsening is allowed only when every sibling is a leaf;
- optional balancing leaves no face-neighbour level jump larger than one.

The normalized coverage test uses exact rational arithmetic. A level-`l` cell in
`d` dimensions has measure `1 / 2**(d*l)`, so the total must equal exactly one.

## 8. What this version intentionally does not solve

This is a topology and geometry baseline, not yet a CFD mesh implementation. It
does not include:

- scalar or vector fields and their transfer between meshes;
- one-flux-per-face numerical connectivity;
- cut cells or curved boundaries;
- PDE discretization or time integration;
- parallel ownership and ghost cells;
- GPU or differentiable tree mutation.

Those features should be added one at a time, with the simple tree remaining a
reference against which optimized versions can be tested.
