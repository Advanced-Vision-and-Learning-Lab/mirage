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

Mirage supports `paligemma`, `qwen2vl`, `gemma4-e4b`, `gemma4-12b`, `ministral3-8b`, `ministral3-14b`, and `aya-vision-8b`, plus `-think` variants of the Gemma 4 keys for a thinking ablation. See the [documentation](https://mirage-f432d87b.mintlify.app) for dataset setup, metrics, model configuration, and HPC instructions.

## Reference results

Scores are percentages. PaliGemma-3B and Qwen2-VL rows use 1,000 images per dataset; Gemma 4, Ministral 3, and Aya Vision rows use 500 images, a prefix of the same candidate pool. THRONE F1 uses a FLAN-T5-XL evaluator with 38 distractors per ground-truth object for every row and supersedes the earlier FLAN-T5-base values. ROPE strict accuracy depends on answer format; see the [documentation](https://mirage-f432d87b.mintlify.app) for notes on individual models. Higher is better for POPE, ROPE, and THRONE; lower is better for CHAIR-I.

| Dataset | Model | POPE accuracy ↑ | ROPE strict accuracy ↑ | CHAIR-I hallucination rate ↓ | THRONE F1 ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| COCO | PaliGemma-3B | 94.2% | 21.6% | 9.1% | 80.9% |
| COCO | Qwen2-VL-2B | 93.6% | 3.6% | 12.8% | 83.8% |
| COCO | Qwen2-VL-7B | 93.5% | 13.1% | 12.7% | 84.4% |
| COCO | Gemma-4-E4B | 93.5% | 0.0% | 12.5% | 77.0% |
| COCO | Ministral-3-8B | 88.1% | 5.1% | 13.7% | 80.8% |
| COCO | Aya-Vision-8B | 95.5% | 24.5% | 16.9% | 82.2% |
| COCO | Gemma-4-12B | 93.0% | 44.4% | 15.2% | 78.4% |
| COCO | Ministral-3-14B | 88.2% | 3.0% | 14.1% | 81.6% |
| Visual Genome | PaliGemma-3B | 88.0% | 8.2% | 67.8% | 27.6% |
| Visual Genome | Qwen2-VL-2B | 88.7% | 8.6% | 75.6% | 27.4% |
| Visual Genome | Qwen2-VL-7B | 89.5% | 10.3% | 69.3% | 27.2% |
| Visual Genome | Gemma-4-E4B | 89.6% | 0.0% | 84.1% | 25.5% |
| Visual Genome | Ministral-3-8B | 84.6% | 0.3% | 87.7% | 26.4% |
| Visual Genome | Aya-Vision-8B | 88.1% | 8.9% | 85.6% | 26.8% |
| Visual Genome | Gemma-4-12B | 90.4% | 21.1% | 76.8% | 27.1% |
| Visual Genome | Ministral-3-14B | 83.9% | 0.1% | 86.6% | 26.6% |

The [results CSV](docs/model-results.csv) is the source for these values.
