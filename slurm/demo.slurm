#!/usr/bin/env bash
# Copy this file to your cluster and edit the site-specific resource and module
# settings. Submit from the project root with: sbatch slurm/demo.slurm
# Prepare datasets and model weights before submitting when the cluster is offline.

#SBATCH --job-name=mirage-demo
#SBATCH --output=mirage_%j.out
#SBATCH --error=mirage_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail

PROJECT_ROOT="${MIRAGE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"

# Replace these with the module names provided by your cluster, if needed.
# module load Python/<version>
# module load CUDA/<version>

if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
elif [[ -f "${PROJECT_ROOT}/venv/bin/activate" ]]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
fi

export HF_HOME="${MIRAGE_HF_HOME:-${PROJECT_ROOT}/huggingface}"
export MIRAGE_VL_MODEL="${MIRAGE_VL_MODEL:-paligemma}"
export MIRAGE_CACHE_DIR="${MIRAGE_CACHE_DIR:-${PROJECT_ROOT}/cache/${SLURM_JOB_ID:-local}}"

python demo.py \
    --dataset "${MIRAGE_DATASET:-coco}" \
    --metric "${MIRAGE_METRIC:-chair}" \
    --model "${MIRAGE_VL_MODEL}" \
    --samples "${MIRAGE_SAMPLES:-5}" \
    --output "${MIRAGE_OUTPUT:-results/demo_${SLURM_JOB_ID:-local}.json}"
