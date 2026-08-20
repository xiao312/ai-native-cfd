# Provenance of the remembered OpenFOAM dolphin

Checked on 2026-08-20.

## Conclusion

The strongest match is `Dolphin.gambit.gz` from a March 2005 CFD Online thread
called [Harpoon to Foam](https://www.cfd-online.com/Forums/openfoam-meshing/61977-harpoon-foam.html).
It was a community mesh-conversion test, not an official OpenFOAM tutorial case.

The distinction is easy to miss because the pasted OpenFOAM log contains a path
under `run/tutorials/icoFoam`. That says where the user launched the converter; it
does not establish that the dolphin was distributed in `$FOAM_TUTORIALS`.

## What the historical thread establishes

- A user was evaluating the commercial Harpoon 1.4.0 cut-hex mesher with
  OpenFOAM 1.0.2.
- The user shared a roughly 782 kB file named `Dolphin.gambit.gz`, which had been
  provided by someone else.
- Its header reported 30,683 nodes and 32,764 elements, with zero boundary sets.
- `gambitToFoam` crashed while reading it. Follow-up posts discuss header and
  boundary-set problems, hanging nodes, and the desirability of a native OpenFOAM
  export.
- The discussion continued into 2006, when Harpoon added direct OpenFOAM export.

That history is technically relevant to this project: it is an early example of
Cartesian/cut-hex meshing meeting a general polyhedral CFD representation. It is
not, however, a reproducible open tutorial by modern standards.

## What the search did not establish

Searches of current public OpenFOAM documentation, tutorial indexes, and source
repositories did not locate a bundled dolphin case. Absence from current trees does
not prove that no independent old tutorial ever used one, but no evidence found so
far supports that origin.

Current official [`snappyHexMesh` documentation](https://doc.openfoam.com/2606/tools/pre-processing/mesh/generation/snappyhexmesh/)
uses `iglooWithFridges` to illustrate castellation, snapping, and layer addition.
That is a documented reference for the workflow; it is unrelated to the 2005
dolphin file.

## Repository decision

Do not vendor the historical Gambit file. Its original university-hosted download
is no longer available through the thread, its authorship is indirect, and no
license was stated.

If we later want a dolphin-shaped 3D importer test, Wikimedia Commons hosts a
separate [`Dolphin.stl`](https://commons.wikimedia.org/wiki/File:Dolphin.stl) by
John Burkardt under CC BY 3.0. It has clear attribution metadata, but there is no
evidence that it is the same geometry as the 2005 Harpoon mesh. We should record
its attribution and license if we choose to download it in a future 3D milestone.

For the present 2D milestone, generated circles and NACA sections remain better
verification geometries because their scale, curvature, area, and expected local
refinement are controlled directly in the tests.
