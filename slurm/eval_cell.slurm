#!/usr/bin/env bash
# One evaluation cell (model x dataset x metric).
# Settings come from the environment, so a launcher can submit many cells:
#
#   MIRAGE_VL_MODEL=qwen2vl MIRAGE_DATASET=coco MIRAGE_METRIC=pope MIRAGE_SAMPLES=500 \
#   MIRAGE_CACHE_DIR=$PWD/cache/qwen2vl_coco_pope \
#       sbatch --time=01:00:00 --job-name=qwen2vl_coco_pope slurm/eval_cell.slurm
#
# Give each cell its own MIRAGE_CACHE_DIR (one SQLite writer per file). A resubmitted
# cell resumes from that cache. THRONE reuses the captions cell's cache dir. --cache-only
# freezes both the caption and the evaluator caches (a miss raises), so use it to replay
# a scored THRONE cell; a new THRONE cell must omit it so the evaluator can fill its cache.
# Set MIRAGE_<KEY>_MODEL_PATH to an HF snapshot path (.../snapshots/<sha>) so cache
# keys pin the model revision; a bare model id is keyed as "unpinned".
# CPU-only cache replays: sbatch --partition=short --gres=none ...
# Create slurm/logs/ before the first submission.

#SBATCH --job-name=mirage-cell
#SBATCH --account=<your-account>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/%x_%j.out

set -euo pipefail

PROJECT_ROOT="${MIRAGE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"

module load GCC/13.3.0 Python/3.12.3 CUDA/12.8.0
source "${PROJECT_ROOT}/venv/bin/activate"

export HF_HOME="${MIRAGE_HF_HOME:-${PROJECT_ROOT}/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
: "${MIRAGE_CACHE_DIR:?set MIRAGE_CACHE_DIR to the cache directory for this cell}"
export MIRAGE_CACHE_DIR

name="${SLURM_JOB_NAME:-local}"
# shellcheck disable=SC2086  # MIRAGE_EXTRA_ARGS is a list of flags
python run.py \
    --model "${MIRAGE_VL_MODEL:?set MIRAGE_VL_MODEL}" \
    --dataset "${MIRAGE_DATASET:?set MIRAGE_DATASET}" \
    --metric "${MIRAGE_METRIC:?set MIRAGE_METRIC}" \
    --samples "${MIRAGE_SAMPLES:-500}" \
    --output "${MIRAGE_OUTPUT:-results/${name}_${SLURM_JOB_ID:-local}.json}" \
    ${MIRAGE_EXTRA_ARGS:-}
