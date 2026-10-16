# Mirage

Mirage evaluates object hallucination in vision-language models using COCO and Visual Genome. It includes metric implementations, local model wrappers, and prediction caching.

**Documentation:** [mirage-f432d87b.mintlify.app](https://mirage-f432d87b.mintlify.app)

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
pytest -q tests/test_demo_smoke.py
```

## Run an evaluation

Prepare the dataset files and model weights locally, then run:

```bash
python demo.py --dataset coco --metric chair --model paligemma --samples 5 --output results/chair-coco.json
```

Mirage supports `paligemma`, `qwen2vl`, `gemma4-e4b`, `gemma4-12b`, `ministral3-8b`, `ministral3-14b`, `aya-vision-8b`, and `muse-glimmer-30b`, plus `-think` variants of the Gemma 4 keys for a thinking ablation. See the [documentation](https://mirage-f432d87b.mintlify.app) for dataset setup, metrics, model configuration, and HPC instructions.

## Reference results

Placeholder: headline values only, pending the homepage redesign. Scores are percentages on the shared first 500 images per dataset; higher is better. Each column is a paper metric with named modifications for sparse labels, scored from cached outputs and static judge verdicts; see [methodology](https://mirage-f432d87b.mintlify.app/methodology) for the definitions and the raw → Mod comparison.

| Dataset | Model | Mod-POPE balanced accuracy ↑ | Mod-ROPE accuracy ↑ | Mod-CHAIR F0.5 ↑ | THRONE F0.5 ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| COCO | Muse-Glimmer-30B | 91.9% | 70.0% | 89.2% | 82.5% |
| COCO | Gemma-4-12B | 92.6% | 74.8% | 84.2% | 79.0% |
| COCO | Gemma-4-E4B | 93.9% | 57.7% | 83.2% | 78.0% |
| COCO | Aya-Vision-8B | 95.6% | 59.2% | 84.2% | 78.8% |
| COCO | Ministral-3-14B | 86.6% | 36.3% | 82.9% | 78.2% |
| COCO | Ministral-3-8B | 86.4% | 26.6% | 82.0% | 78.6% |
| COCO | Qwen2-VL-7B | 93.1% | 38.1% | 85.1% | 82.1% |
| COCO | Qwen2-VL-2B | 93.1% | 10.7% | 84.8% | 82.9% |
| COCO | PaliGemma-3B | 94.0% | 41.4% | 81.7% | 78.7% |
| Visual Genome | Muse-Glimmer-30B | 90.7% | 64.4% | 79.6% | 73.9% |
| Visual Genome | Gemma-4-12B | 93.6% | 65.5% | 70.5% | 71.4% |
| Visual Genome | Gemma-4-E4B | 92.8% | 48.9% | 63.8% | 68.9% |
| Visual Genome | Aya-Vision-8B | 92.0% | 46.2% | 73.8% | 71.7% |
| Visual Genome | Ministral-3-14B | 85.8% | 35.8% | 69.3% | 69.1% |
| Visual Genome | Ministral-3-8B | 87.3% | 24.7% | 70.8% | 68.5% |
| Visual Genome | Qwen2-VL-7B | 92.6% | 50.6% | 43.6% | 65.9% |
| Visual Genome | Qwen2-VL-2B | 91.6% | 43.8% | 49.8% | 68.7% |
| Visual Genome | PaliGemma-3B | 92.0% | 27.0% | 50.0% | 63.0% |

The [results CSV](docs/model-results.csv) is the source for these values.
