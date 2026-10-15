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

Mirage supports `paligemma`, `qwen2vl`, `gemma4-e4b`, `gemma4-12b`, `ministral3-8b`, `ministral3-14b`, and `aya-vision-8b`. See the [documentation](https://mirage-f432d87b.mintlify.app) for dataset setup, metrics, model configuration, and HPC instructions.

## Reference results

Scores are percentages from 1,000 images per model and dataset. Higher is better for POPE, ROPE, and THRONE; lower is better for CHAIR-I.

| Dataset | Model | POPE accuracy ↑ | ROPE strict accuracy ↑ | CHAIR-I hallucination rate ↓ | THRONE F1 ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| COCO | PaliGemma-3B | 94.2% | 21.6% | 9.1% | 72.9% |
| COCO | Qwen2-VL-2B | 93.6% | 3.6% | 12.8% | 73.4% |
| COCO | Qwen2-VL-7B | 93.5% | 13.1% | 12.7% | 72.9% |
| Visual Genome | PaliGemma-3B | 88.0% | 8.2% | 67.8% | 21.5% |
| Visual Genome | Qwen2-VL-2B | 88.7% | 8.6% | 75.6% | 24.0% |
| Visual Genome | Qwen2-VL-7B | 89.5% | 10.3% | 69.3% | 23.6% |

The [results CSV](docs/model-results.csv) is the source for these values.
