#!/usr/bin/env bash
# Run the fixed-charge benchmark under NVIDIA Nsight Systems on a GPU compute node.
# Profiles the TorchFF path where perf_op is called (bond, angle, nonbonded, long-range).
#
# Usage:
#   ./benchmark/run_profile_slurm.sh [optional nsys output stem]
#
# Example:
#   ./benchmark/run_profile_slurm.sh
#   ./benchmark/run_profile_slurm.sh my_profile
#
# Output: <stem>.nsys-rep in the benchmark/ directory. View with:
#   nsys-ui benchmark/<stem>.nsys-rep
# or export to CSV/other formats with nsys export.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

STEM="${1:-water_md_nve_profile}"
OUTPUT_DIR="$(pwd)"
OUTPUT_REP="${OUTPUT_DIR}/${STEM}.nsys-rep"
OUTPUT_SQLITE="${OUTPUT_DIR}/${STEM}.sqlite"
REPLAY_KERNELS_TXT="${OUTPUT_DIR}/${STEM}_replay_kernels.txt"

# Force overwrite: remove existing report artifacts so this run writes fresh files.
rm -f "$OUTPUT_REP" "$OUTPUT_SQLITE" "${OUTPUT_DIR}/${STEM}_replay_kernels.txt" "${OUTPUT_DIR}/${STEM}.txt"

echo "Profiling TorchFF fixed-charge benchmark with Nsight Systems"
echo "Output report: ${OUTPUT_REP}"
echo "Run with srun on a GPU node (conda env openmm-torch-py312-cu124 per .cursor/rules/torchff-project-and-testing.mdc)..."

# Default: 1 node, premium QOS, 1 hour, GPU. Override with env if needed.
SRUN_QOS="${SRUN_QOS:-premium}"
SRUN_TIME="${SRUN_TIME:-1:00:00}"
SRUN_ACCOUNT="${SRUN_ACCOUNT:-m2834}"

# Conda env per torchff-project-and-testing.mdc (openmm-torch-py312-cu124)
CONDA_ENV_NAME="openmm-torch-py312-cu124"

# On the compute node: load conda, activate project env, load CUDA, then profile.
# --gres=gpu:1 ensures Slurm assigns a GPU and sets CUDA_VISIBLE_DEVICES so PyTorch sees it.
srun --nodes 1 --qos "$SRUN_QOS" --time "$SRUN_TIME" --constraint gpu --gres=gpu:1 --account="$SRUN_ACCOUNT" \
  bash -c '
    module load conda
    mamba activate '"$CONDA_ENV_NAME"'
    module load cudatoolkit 2>/dev/null || true
    export CMM_PROFILE=1
    nsys profile \
      --force-overwrite=true \
      -o "'"$OUTPUT_REP"'" \
      --stats=true \
      --trace=cuda,nvtx,cudnn,cublas \
      --cuda-graph-trace=node \
      python water_md_nve.py
  '

echo "Generating kernel summary for NVTX range 'perf_op replay (batch)'..."
nsys stats "$OUTPUT_REP" \
  --report cuda_gpu_kern_sum \
  --filter-nvtx "CMM Step" \
  --format table \
  --force-export=true \
  --force-overwrite=true \
  2>/dev/null | tee "$REPLAY_KERNELS_TXT"

echo "Done. Report: ${OUTPUT_REP}"
echo "Kernels in 'perf_op replay (batch)': ${REPLAY_KERNELS_TXT}"
echo "View with: nsys-ui ${OUTPUT_REP}"
echo "Or export: nsys export --type=json ${OUTPUT_REP} ${OUTPUT_REP%.nsys-rep}.json"
