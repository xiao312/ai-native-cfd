#!/usr/bin/env bash
#SBATCH --job-name=aicfd-cavity64
#SBATCH --partition=kshcnormal02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

set -euo pipefail

export LC_ALL=C
export LANG=C

# This order is intentional. The OpenFOAM module sets LD_LIBRARY_PATH, after
# which the compiler and MPI modules add the compatible C++ and MPI runtimes.
module purge
module load openfoam-v7-gcc9.3.0_openmpi4.1.5
module load compiler/gcc/9.3.0
module load mpi/openmpi/gcc-9.3.0/4.1.5

repository_root="${AICFD_REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?submit from the repository root}}"
export AICFD_CAVITY_SOURCE_CASE="$FOAM_TUTORIALS/incompressible/icoFoam/cavity/cavity"
export AICFD_CLUSTER_NAME=kunshan
exec "$repository_root/scripts/openfoam/run_cavity64.sh"
