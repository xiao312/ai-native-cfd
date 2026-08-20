# Visualizing an adaptive geometry discretization

The visualization module answers a basic debugging question:

> Did the tree refine where we expected, and did clipping attach the correct
> geometry data to every current leaf?

It writes a self-contained SVG image. SVG is useful at this stage because a web
browser can open it, the file remains sharp when zoomed, and Python can generate it
without installing a plotting package. This is a geometry diagnostic, not a CFD
post-processor.

## 1. Keep representation separate from presentation

The dependency direction is one way:

```text
AdaptiveTree + Obstacle2D + SnappedGeometry2D
                      |
                      v
                 SVG renderer
```

The tree and clipping code do not know that SVG exists. The renderer only reads
their public data. This lets us later add Matplotlib, PyVista, VTK, or a browser UI
without changing the numerical representation.

The minimal API is:

```python
from aicfd.visualization import SvgOptions, write_geometry_svg

write_geometry_svg(
    "mesh.svg",
    tree,
    obstacle=obstacle,
    snapped=snapped,
    options=SvgOptions(show_snapped_points=True, show_normals=True),
)
```

`render_geometry_svg(...)` returns the same image as a string when a caller wants
to store or serve it in another way.

## 2. The drawing is built in layers

Back to front, the renderer draws:

1. one rectangle for every current quadtree leaf;
2. the input obstacle polygon;
3. the exact clipped boundary fragment inside each cut cell;
4. optional cell-edge/obstacle intersection points;
5. optional outward boundary-normal arrows;
6. a color legend.

The rectangles remain Cartesian because our first "snapping" algorithm is an
embedded-boundary method. The orange fragment and point overlays show where the
physical boundary actually cuts those rectangles; they do not imply that a tree
vertex has moved.

## 3. Color is a diagnostic quantity

`SvgOptions.color_by` supports four modes:

| Mode | Meaning | Needs snapped data? |
| --- | --- | --- |
| `none` | transparent cells; show topology only | no |
| `level` | deeper refinement uses darker blue | no |
| `classification` | fluid, cut, or solid | yes |
| `fluid_fraction` | grey at `chi=0`, white at `chi=1` | yes |

Level coloring helps find unexpected refinement and coarse-fine transitions.
Classification coloring checks the topology of the embedded obstacle. Fluid
fraction coloring exposes tiny cut-cell slivers that may later restrict an explicit
time step.

## 4. Mapping physical coordinates to the screen

SVG's horizontal coordinate increases to the right, as physical `x` usually does.
Its vertical coordinate increases downward, opposite to physical `y`. The renderer
therefore maps a physical point `(x, y)` to

```text
screen_x = margin + scale_x * (x - origin_x)
screen_y = margin + plot_height - scale_y * (y - origin_y)
```

The same physical scale is used in both directions, so a physical circle remains a
circle even when the domain is rectangular.

## 5. The SVG is also inspectable data

Each cell is a group with attributes such as:

```xml
<g class="cell"
   data-cell-id="2D:L4:M37"
   data-level="4"
   data-classification="cut"
   data-fluid-fraction="0.731...">
```

Hovering a cell in many SVG viewers displays the same values as a tooltip. Scripts
can also parse these attributes. The SVG is still only a diagnostic export: the
authoritative values remain the Python tree and snapped-geometry objects.

## 6. Snapshots must match the current leaves

`SnappedGeometry2D` describes the leaf set at the instant clipping was performed.
If the tree is refined or coarsened afterward, that snapshot is stale. Rendering it
would silently attach values to the wrong mesh, so the renderer compares the two
leaf sets and raises an error. Re-run `snap_to_obstacle` after every tree mutation.

## 7. Run the lightweight examples

No flow equation or OpenFOAM process is run by these commands:

```bash
uv run python examples/geometry_amr.py --shape circle --output circle.svg
uv run python examples/geometry_amr.py --shape naca0012 \
  --color-by fluid_fraction --show-points --show-normals \
  --output naca0012-debug.svg
```

The current backend is intentionally limited to 2D. A later 3D path should export
octree cells and clipped surfaces to VTK/VTU for ParaView instead of trying to make
SVG act like a 3D format.
