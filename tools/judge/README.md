# Judge

Grades the items that Mirage's judged metrics need: caption mentions (Mod-CHAIR, THRONE
completed labels), ROPE answers (Mod-ROPE) and POPE no-question objects (Mod-POPE drop list).
Scoring never runs this tool; it reads the verdicts it produced from `judgments/`.

| Skill | Version | SHA-256 prefix | Verdicts |
| --- | --- | --- | --- |
| `judge-mentions` | rubric v3.1, skill version 2 | `fcbf47e81228` | present, absent, not_object, unsure |
| `judge-rope` | rubric v1, skill version 1 | `0835e4524094` | correct, related, wrong, unsure |
| `judge-pope` | rubric v1, skill version 1 | `fa3c04727715` | present, absent, unsure |

The rubrics are the `SKILL.md` files under `.claude/skills/`. Every run used `claude-opus-5-5`.

## Run

`run_judge.py` sends each input chunk to a fresh headless `claude -p` session with the skill,
validates the output, retries once, and writes `<chunk>.out.json` plus a `.meta.json` with the
skill hash, model, usage and cost. Finished chunks are skipped, so a run resumes.

    python3 run_judge.py --skill judge-mentions --out runs/new chunks/*.json
    python3 run_judge.py --export runs/new >> rows.tsv

Input chunks are JSON lists of 2–10 images, each with `image_id` (`<dataset>_<id>`) and
`image` (a local path), plus one of:

- `mentions`: `[{"model", "word", "context"}]`, the sentence the word appears in;
- `boxes`: `[{"box", "answers": [{"answer", "example"}]}]`, on an image with numbered boxes drawn;
- `objects`: `["name", ...]`, the POPE no-question objects.

Exported rows are `item<TAB>key<TAB>verdict` with keys `<model>|<word>` or `<box>|<answer>`,
the format of `judgments/*.tsv.gz`. POPE verdicts become drop-list rows (present or unsure;
junk names are dropped without the judge). `python -m metrics.judged` lists the items that
have no verdict yet.
