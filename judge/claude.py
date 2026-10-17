"""Live judge: run a judge skill over input chunks with headless `claude -p`, one fresh session per chunk.

Off unless asked for (``Judge.load(live=True)``). Resumable: a chunk whose output already
validates against its input is skipped. Each chunk also gets a <stem>.meta.json sidecar with
the model, skill hash, usage and cost. The binary is ``$MIRAGE_CLAUDE_BIN``, else ``claude`` on PATH.
The model defaults to the manifest's "judge_model", and a skill whose hash differs from the
manifest's "skills" is refused.

    python -m judge.claude --skill judge-mentions --out runs/new in_*.json
    python -m judge.claude --export runs/new in_*.json >> rows.tsv   # verdict rows for judgments/*.tsv.gz
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

from .store import MANIFEST

HERE = Path(__file__).resolve().parent
SPEC = json.loads(MANIFEST.read_text())

SKILLS = {"judge-mentions": {"present", "absent", "not_object", "unsure"},
          "judge-rope": {"correct", "related", "wrong", "unsure"},
          "judge-pope": {"present", "absent", "unsure"}}


def expected_keys(chunk):
    if "objects" in chunk[0]:
        return {e["image_id"]: set(e["objects"]) for e in chunk}
    if "mentions" in chunk[0]:
        return {e["image_id"]: {f'{m["model"]}|{m["word"]}' for m in e["mentions"]} for e in chunk}
    return {e["image_id"]: {f'{b["box"]}|{a["answer"]}' for b in e["boxes"] for a in b["answers"]} for e in chunk}


def problems(chunk, out_path, verdicts):
    try:
        out = json.loads(Path(out_path).read_text())
    except Exception as e:  # missing or not JSON
        return [f"unreadable: {e}"]
    errs = []
    for iid, keys in expected_keys(chunk).items():
        got = out.get(iid, {})
        missing = keys - set(got)
        bad = {k: v for k, v in got.items() if v not in verdicts}
        if missing:
            errs.append(f"{iid}: {len(missing)} missing")
        if bad:
            errs.append(f"{iid}: bad verdicts {bad}")
    return errs


def claude_binary(binary=None):
    """``binary``, else ``$MIRAGE_CLAUDE_BIN``, else ``claude`` on PATH."""
    binary = binary or os.environ.get("MIRAGE_CLAUDE_BIN") or shutil.which("claude")
    if not binary:
        raise FileNotFoundError("no claude binary: set MIRAGE_CLAUDE_BIN or put claude on PATH")
    return binary


def skill_sha(skill):
    return hashlib.sha256((HERE / f".claude/skills/{skill}/SKILL.md").read_bytes()).hexdigest()[:12]


def run_chunk(inp, out, model, timeout, skill, binary="claude"):
    chunk = json.loads(Path(inp).read_text())
    if any(c.isspace() for c in f"{Path(inp).resolve()}{Path(out).resolve()}"):
        raise ValueError(f"the /{skill} invocation is space-separated; paths with whitespace break it: {inp}, {out}")
    dirs = {str(Path(inp).resolve().parent), str(Path(out).resolve().parent)}
    dirs |= {str(Path(e["image"]).resolve().parent) for e in chunk}
    cmd = [binary, "-p", f"/{skill} {Path(inp).resolve()} {Path(out).resolve()}",
           "--model", model, "--output-format", "json",
           "--allowedTools", "Read", "Write",
           "--setting-sources", "project", "--strict-mcp-config", "--no-session-persistence"]
    for d in sorted(dirs):
        cmd += ["--add-dir", d]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:  # a failed attempt, like a nonzero exit
        return None, time.time() - t0, {}, f"timed out after {timeout}s", chunk
    wall = time.time() - t0
    try:
        res = json.loads(proc.stdout)
    except json.JSONDecodeError:
        res = {"raw_stdout": proc.stdout[-2000:]}
    return proc.returncode, wall, res, proc.stderr[-2000:], chunk


def run_chunks(inputs, outdir, skill, model=None, timeout=1800, retries=1, binary="claude"):
    """Grade each input chunk into ``outdir``; returns the total reported cost in USD.

    Raises:
        ValueError: The skill's hash differs from the manifest's (its verdicts would mix rubrics).
    """
    model = model or SPEC["judge_model"]
    verdicts = SKILLS[skill]
    skill_hash, frozen_hash = skill_sha(skill), SPEC["skills"][skill.removeprefix("judge-")]
    if skill_hash != frozen_hash:
        raise ValueError(f"{skill} SKILL.md hash {skill_hash} differs from the manifest's {frozen_hash}")
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    total_cost = 0.0
    for inp in inputs:
        stem = Path(inp).stem
        out = outdir / f"{stem}.out.json"
        chunk = json.loads(Path(inp).read_text())
        if not problems(chunk, out, verdicts):
            print(f"{stem}: done already, skipping", flush=True)
            continue
        for attempt in range(1 + retries):
            rc, wall, res, err, chunk = run_chunk(inp, out, model, timeout, skill, binary)
            errs = problems(chunk, out, verdicts)
            meta = {"input": str(Path(inp).resolve()), "skill": skill, "model": model, "skill_sha": skill_hash,
                    "attempt": attempt, "returncode": rc, "wall_s": round(wall, 1),
                    "cost_usd": res.get("total_cost_usd"), "usage": res.get("usage"),
                    "num_turns": res.get("num_turns"), "result": str(res.get("result", ""))[:500],
                    "valid": not errs, "problems": errs[:20], "stderr": err}
            (outdir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=1))
            total_cost += res.get("total_cost_usd") or 0
            print(f"{stem}: attempt {attempt} rc={rc} {wall:.0f}s cost={res.get('total_cost_usd')} valid={not errs} {errs[:3]}", flush=True)
            if not errs:
                break
    return total_cost


def export(inputs, outdir):
    """``(item, key, verdict)`` rows of the chunks ``inputs`` from their outputs in ``outdir``,
    only the keys each chunk asked for."""
    rows = []
    for inp in inputs:
        got = json.loads((Path(outdir) / f"{Path(inp).stem}.out.json").read_text())
        want = expected_keys(json.loads(Path(inp).read_text()))
        rows += [(item, k, got[item][k]) for item, keys in want.items() for k in sorted(keys) if k in got.get(item, {})]
    return rows


class ClaudeCLI:
    """The live backend of :class:`judge.Judge`: grades requests with the skills in judge/.claude/skills.

    Args:
        model: Claude model the skill runs on (default: the manifest's "judge_model").
        binary: The claude executable; default ``$MIRAGE_CLAUDE_BIN``, else ``claude`` on PATH.
        workdir: Where chunks and outputs go (default: a new temporary directory per call); pass
            the directory of a failed call to resume it.
        chunk: Images per ``claude -p`` session.
    """

    def __init__(self, model=None, binary=None, workdir=None, chunk=10, timeout=1800, retries=1):
        self.model, self.workdir, self.chunk, self.timeout, self.retries = (
            model or SPEC["judge_model"], workdir, chunk, timeout, retries)
        self.binary = claude_binary(binary)

    def grade(self, kind, requests):
        """Verdict rows ``(item, key, verdict)`` for skill inputs ``requests`` (one per image).

        Raises:
            ValueError: The skill's hash differs from the manifest's.
            RuntimeError: A chunk still fails validation after the retries (errors name the work directory).
        """
        skill = f"judge-{kind}"
        if self.workdir:
            work = Path(self.workdir); work.mkdir(parents=True, exist_ok=True)
        else:
            work = Path(tempfile.mkdtemp(prefix=f"mirage-{skill}-"))
        inputs = []
        for c in range(0, len(requests), self.chunk):
            inputs.append(work / f"in_{c // self.chunk:04d}.json")
            inputs[-1].write_text(json.dumps(requests[c:c + self.chunk], indent=1))
        try:
            run_chunks(inputs, work / "out", skill, self.model, self.timeout, self.retries, self.binary)
        except Exception as e:
            e.add_note(f"{skill} work directory (pass it as workdir to resume): {work}")
            raise
        bad = [p.name for p in inputs if problems(json.loads(p.read_text()), work / "out" / f"{p.stem}.out.json",
                                                  SKILLS[skill])]
        if bad:
            raise RuntimeError(f"{skill}: {len(bad)} chunks failed validation in {work}: {bad[:5]}")
        return export(inputs, work / "out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--export", help="print the verdicts of the input chunks in this run directory as "
                                     "item<TAB>key<TAB>verdict rows")
    ap.add_argument("--model")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--skill", default="judge-mentions", choices=sorted(SKILLS))
    a = ap.parse_args()
    if not a.inputs:
        ap.error("input chunks are required")
    if a.export:
        print("".join(f"{item}\t{k}\t{v}\n" for item, k, v in export(a.inputs, a.export)), end="")
        return
    if not a.out:
        ap.error("--out is required")
    total_cost = run_chunks(a.inputs, a.out, a.skill, a.model, a.timeout, a.retries, claude_binary())
    print(f"total reported cost this run: {total_cost:.2f} USD", flush=True)


if __name__ == "__main__":
    sys.exit(main())
