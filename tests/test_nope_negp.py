"""Acceptance tests for NOPE NegP detection (metrics.nope.is_negp).

Runs fully offline (no GPU/torch). Pins the word-boundary contract: every NegP
answer must match as a standalone token, never inside a longer word or number,
so hallucinating answers like "There are 10 dogs." no longer score as correct
negations in negp_accuracy.

Run:  python3 -m pytest tests/test_nope_negp.py -q
      (or plain `python3 tests/test_nope_negp.py` for a no-pytest smoke run)
"""
import json
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _import_nope():
    # metrics/__init__ imports torch; load the torch-free metrics.nope without it.
    if "metrics" not in sys.modules:
        pkg = types.ModuleType("metrics")
        pkg.__path__ = [str(ROOT / "metrics")]
        sys.modules["metrics"] = pkg
    from metrics import nope
    return nope


# Substring false positives: "0" inside "10"/"20", "no" inside "snowman"/"unknown"/"snow".
NOT_NEGP = [
    "There are 10 dogs.",
    "It is a snowman.",
    "The answer is unknown to me.",
    "Yes, there is snow in the image",
    "20 people",
]

NEGP = [
    "No",
    "no.",
    "None",
    "There is no one here",
    "nobody",
    "zero",
    "0",
    "Neither is present",
    "No, there is no dog in the image",
]


def test_substring_matches_are_not_negp():
    is_negp = _import_nope().is_negp
    for response in NOT_NEGP:
        assert not is_negp(response), f"{response!r} must not count as a negation"


def test_standalone_negp_answers_still_match():
    is_negp = _import_nope().is_negp
    for response in NEGP:
        assert is_negp(response), f"{response!r} must count as a negation"


def test_every_negp_answer_matches_itself():
    nope = _import_nope()
    for answer in nope.NEGP_ANSWERS:
        assert nope.is_negp(answer), f"{answer!r} must match as a bare answer"
        assert nope.is_negp(f"Answer: {answer}."), f"{answer!r} must match when punctuated"


def test_negp_accuracy_excludes_substring_matches():
    """is_negp feeds negp_accuracy, so the fix must move the headline metric."""
    nope = _import_nope()
    records = [{"key": f"{i}|dog", "question": "How many dogs?", "answer": "none"}
               for i in range(len(NOT_NEGP) + 1)]
    with tempfile.TemporaryDirectory() as tmp:
        turn2 = Path(tmp) / "nope_turn2_results.json"
        turn2.write_text(json.dumps(records))
        metric = nope.NOPEMetric(turn2_path=str(turn2))
        results = metric.evaluate(NOT_NEGP + ["None"])
    assert results["negp_accuracy"] == 1 / len(records)
    assert results["standard_accuracy"] == 1 / len(records)
    assert results["hallucination_gap"] == 0.0


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run()
