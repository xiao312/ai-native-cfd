#!/usr/bin/env bash

# Build and run the cavity64 reference case after a cluster wrapper has loaded
# a working OpenFOAM Foundation v7 environment.

set -euo pipefail

export LC_ALL=C
export LANG=C

job_id="${SLURM_JOB_ID:?SLURM_JOB_ID is required; submit through a Slurm wrapper}"
run_root="${AICFD_RUN_ROOT:-$HOME/xk/ai-native-cfd-runs}"
run_directory="$run_root/cavity64-$job_id"
source_case="${AICFD_CAVITY_SOURCE_CASE:?AICFD_CAVITY_SOURCE_CASE is required}"

for executable in \
    foamDictionary blockMesh checkMesh postProcess icoFoam foamListTimes; do
    if ! command -v "$executable" >/dev/null 2>&1; then
        printf 'Required OpenFOAM executable is unavailable: %s\n' "$executable" >&2
        exit 1
    fi
done

if [[ ! -d "$source_case" ]]; then
    printf 'OpenFOAM cavity tutorial is missing: %s\n' "$source_case" >&2
    exit 1
fi
if [[ -e "$run_directory" ]]; then
    printf 'Refusing to overwrite existing run directory: %s\n' "$run_directory" >&2
    exit 1
fi

mkdir -p "$run_root"
cp -R "$source_case" "$run_directory"
cd "$run_directory"

foamDictionary system/blockMeshDict -entry blocks -set \
    '(hex (0 1 2 3 4 5 6 7) (64 64 1) simpleGrading (1 1 1))'
foamDictionary system/controlDict -entry deltaT -set 0.00125
foamDictionary system/controlDict -entry writeInterval -set 1
foamDictionary system/controlDict -entry writePrecision -set 12
foamDictionary system/controlDict -entry timePrecision -set 10

blockMesh > log.blockMesh
checkMesh > log.checkMesh
postProcess -func writeCellCentres -time 0 > log.writeCellCentres
icoFoam > log.icoFoam
foamListTimes -withZero > time-index.txt

time_count=$(wc -l < time-index.txt)
if [[ "$time_count" -ne 401 ]]; then
    printf 'Expected 401 time directories, found %s\n' "$time_count" >&2
    exit 1
fi

{
    printf 'openfoam_version=7\n'
    printf 'mesh=64x64x1\n'
    printf 'delta_t=0.00125\n'
    printf 'end_time=0.5\n'
    printf 'time_frames=%s\n' "$time_count"
    printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
    printf 'slurm_partition=%s\n' "${SLURM_JOB_PARTITION:-unknown}"
    printf 'cluster=%s\n' "${AICFD_CLUSTER_NAME:-unknown}"
} > run-metadata.txt

printf 'AICFD_RUN_DIR=%s\n' "$run_directory"
