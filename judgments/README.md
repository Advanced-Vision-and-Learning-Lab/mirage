# Judgments

Static judge outputs read by `judge.VerdictStore` (checked against `judge/manifest.json`; produced
with `judge/claude.py`, which records the rubric versions and skill hashes). Items are
`<dataset>_<image id>` on the shared first 500 images of COCO val2017 and Visual Genome; the pool
covers the nine models in `docs/model-results.csv`.

| File | Rows | Columns |
| --- | ---: | --- |
| `mentions.tsv.gz` | 43,499 | item, `<model>\|<word>`, verdict (`judge-mentions` rubric v3.1) |
| `rope.tsv.gz` | 28,897 | item, `<box>\|<answer>`, verdict (`judge-rope` v1) |
| `pope_drop_list.tsv` | 569 | dataset, image_id, object, reason (junk, present or unsure; `judge-pope` v1) |
| `committee.json` | 1,000 | per item: words the frozen pool resolved present, and its verified present words (`repro.py --export-committee`) |
| `final_tables.json` | | the published table; `python repro.py --root TREE --check` rebuilds and compares it |

Mention tagging uses NLTK 3.9.4 with WordNet 3.0 and inflect 7.5.0; a different WordNet
changes which words resolve without the judge.
