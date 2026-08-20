#!/usr/bin/env bash
#SBATCH --job-name=aicfd-cavity64
#SBATCH --partition=wzacnormal03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00

set -euo pipefail

export LC_ALL=C
export LANG=C

module purge
module load apps/OpenFOAM/7/hpcx-gcc-7.3.1

# The Wuzhen module exposes OpenFOAM's general libraries but omits the serial
# Pstream implementation from LD_LIBRARY_PATH. icoFoam is serial, so use dummy.
openfoam_root=/public/software/apps/OpenFOAM/7/hpcx-2.4.1-gcc-7.3.1/OpenFOAM-7
export LD_LIBRARY_PATH="$openfoam_root/platforms/linux64GccDPInt32Opt/lib/dummy:${LD_LIBRARY_PATH:-}"
export AICFD_CAVITY_SOURCE_CASE="$openfoam_root/tutorials/incompressible/icoFoam/cavity/cavity"
export AICFD_CLUSTER_NAME=wuzhen

repository_root="${AICFD_REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?submit from the repository root}}"
exec "$repository_root/scripts/openfoam/run_cavity64.sh"
