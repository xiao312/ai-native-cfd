# OpenFOAM cavity64 trajectory

This reference case is generated from the OpenFOAM Foundation v7 tutorial at:

```text
$FOAM_TUTORIALS/incompressible/icoFoam/cavity/cavity
```

The source tutorial is copied at run time, so this repository does not duplicate
OpenFOAM-distributed case files. The Slurm script changes only:

| Setting | Tutorial | This case |
| --- | ---: | ---: |
| cells | `20 x 20 x 1` | `64 x 64 x 1` |
| time step | `0.005 s` | `0.00125 s` |
| write interval | every 20 steps | every step |
| write precision | 6 | 12 |
| time precision | 6 | 10 |

The physical domain remains `0.1 m x 0.1 m`, the lid speed remains `1 m/s`, the
kinematic viscosity remains `0.01 m^2/s`, and the end time remains `0.5 s`.
Consequently, the run writes 400 evolved states plus the initial state.

The `64 x 64` grid corresponds exactly to level 6 of our quadtree. This lets the
first importer validate field ordering without interpolation.

Run it on the Wuzhen SCNet account with:

```bash
sbatch scripts/scnet-wuzhen/run_cavity64.slurm
```

The Kunshan fallback is `scripts/agent-scnet/run_cavity64.slurm`. Both wrappers
load the cluster-specific OpenFOAM environment and then call the same case runner,
`scripts/openfoam/run_cavity64.sh`, so numerical settings cannot silently diverge.

The job copies the tutorial into a new, job-specific directory below
`$HOME/xk/ai-native-cfd-runs`; it never modifies the OpenFOAM installation or an
existing run.
