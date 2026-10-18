# Mirage

Mirage is an evaluation framework for vision language models. It measures object hallucination with POPE, ROPE, CHAIR and THRONE, each modified so it still works on datasets with sparse labels such as Visual Genome. Results, methods and guides are at [mirage-eval.com](https://mirage-eval.com).

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng stopwords wordnet
python run.py --dataset coco --metric pope --model gemma4-e4b --samples 10
python repro.py --records tests/data/records.json.gz --check   # reproduce the results table
```
