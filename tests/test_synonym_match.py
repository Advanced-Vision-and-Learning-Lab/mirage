"""Acceptance tests for the synonym-matching layer (issue #24).

Runs fully offline (WordNet corpus only, no GPU/torch). Encodes the *measured*
Phase-0 behavior of the shipped WordNet hops=0 matcher — including the
deliberate rejections (mug/cup, woman/person) that a hypernym hop would wrongly
merge. See Type1_Experiments/synonym_regrade/phase0_spike.md.

Run:  python3 -m pytest tests/test_synonym_match.py -q
      (or plain `python3 tests/test_synonym_match.py` for a no-pytest smoke run)
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from matching import ExactMatcher, SubstringMatcher, WordNetMatcher, SynonymMatcher


def _import_rope_metric():
    # metrics/__init__ imports torch; load the torch-free metrics.rope without it.
    if "metrics" not in sys.modules:
        pkg = types.ModuleType("metrics")
        pkg.__path__ = [str(ROOT / "metrics")]
        sys.modules["metrics"] = pkg
    from metrics.rope import ROPEMetric
    return ROPEMetric

# Pairs the grader SHOULD merge (true synonyms / number variants / same head noun).
MERGE = [
    ("sofa", "couch"),
    ("couch", "sofa"),
    ("bicycle", "bike"),
    ("small rock", "rocks"),
    ("sliced hard-boiled egg", "eggs"),
    ("car", "automobile"),
    ("soccer ball", "sports ball"),
    ("white line", "lines"),
]

# Pairs the grader must NOT merge. Includes hyponym pairs (mug/cup, woman/person)
# that hops=0 rejects on purpose to keep precision at 1.0.
REJECT = [
    ("cat", "dog"),
    ("mug", "cup"),
    ("towel", "couch"),
    ("monitor", "tv"),
    ("woman", "person"),
    ("man", "person"),
    ("bottle", "wine"),
    ("ski boot", "skis"),
]


def test_wordnet_merges_true_synonyms():
    wn = WordNetMatcher()  # hops=0
    for a, b in MERGE:
        assert wn.match(a, b), f"WordNet should merge {a!r}/{b!r}"


def test_wordnet_rejects_hard_negatives():
    wn = WordNetMatcher()  # hops=0
    for a, b in REJECT:
        assert not wn.match(a, b), f"WordNet should NOT merge {a!r}/{b!r}"


def test_synonym_matcher_default_is_offline_and_deterministic():
    syn = SynonymMatcher()  # base + wordnet, no judge/cache
    # Never raises (no MissingPairError) and is repeatable.
    for a, b in MERGE + REJECT:
        assert syn.match(a, b) == syn.match(a, b)


def test_synonym_superset_of_lenient_invariant():
    """overall_accuracy_synonym >= overall_accuracy: synonym matches wherever
    lenient does (synonym includes SubstringMatcher as its base tier)."""
    sub = SubstringMatcher()
    syn = SynonymMatcher()
    cases = MERGE + REJECT + [
        ("wooden cabinet", "cabinet"),   # lenient True -> synonym must be True
        ("red fire hydrant", "hydrant"),
        ("person", "person"),
    ]
    for a, b in cases:
        if sub.match(a, b):
            assert syn.match(a, b), f"synonym must hold wherever lenient does: {a!r}/{b!r}"


def test_exact_and_substring_no_regression():
    """ExactMatcher/SubstringMatcher reproduce the old _rope_match* semantics."""
    ex, sub = ExactMatcher(), SubstringMatcher()
    # exact: case-insensitive equality only
    assert ex.match("Cabinet", "cabinet")
    assert not ex.match("wooden cabinet", "cabinet")
    # lenient: whole-word substring either direction, but respects word boundaries
    assert sub.match("wooden cabinet", "cabinet")
    assert sub.match("cabinet", "wooden cabinet")
    assert not sub.match("cabinet", "cabinets")   # no word boundary -> not a match
    assert not sub.match("", "cabinet")           # empty -> False


def test_rope_metric_inline_synonym_matcher():
    """ROPEMetric(matchers=[..., SynonymMatcher()]) emits the synonym column
    while the lenient/strict headline columns stay untouched."""
    ROPEMetric = _import_rope_metric()
    metric = ROPEMetric(
        num_objects=2,
        matchers=[SubstringMatcher(), ExactMatcher(), SynonymMatcher()],
    )
    # obj1: WordNet-only credit; obj2: lenient (substring) credit.
    metric.evaluate_predictions([["bike", "wooden cabinet"]], [["bicycle", "cabinet"]])
    results = metric.format_results()
    assert results["overall_accuracy_synonym"] == 1.0
    assert results["per_position_accuracy_synonym"] == {"obj1": 1.0, "obj2": 1.0}
    assert results["overall_accuracy"] == 0.5
    assert results["per_position_accuracy"] == {"obj1": 0.0, "obj2": 1.0}
    assert results["overall_accuracy_strict"] == 0.0


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run()
