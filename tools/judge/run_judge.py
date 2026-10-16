"""Run a judge skill over input chunks with headless `claude -p`, one fresh session per chunk.

Resumable: a chunk whose output already validates against its input is skipped.
Each chunk also gets a <stem>.meta.json sidecar with the model, skill hash, usage and cost.

    python3 run_judge.py --skill judge-mentions --out runs/new in_*.json
    python3 run_judge.py --export runs/new >> rows.tsv   # verdict rows for judgments/*.tsv.gz
"""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent

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


def run_chunk(inp, out, model, timeout, skill):
    chunk = json.loads(Path(inp).read_text())
    dirs = {str(Path(inp).resolve().parent), str(Path(out).resolve().parent)}
    dirs |= {str(Path(e["image"]).resolve().parent) for e in chunk}
    cmd = ["claude", "-p", f"/{skill} {Path(inp).resolve()} {Path(out).resolve()}",
           "--model", model, "--output-format", "json",
           "--allowedTools", "Read", "Write",
           "--setting-sources", "project", "--strict-mcp-config", "--no-session-persistence"]
    for d in sorted(dirs):
        cmd += ["--add-dir", d]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=timeout)
    wall = time.time() - t0
    try:
        res = json.loads(proc.stdout)
    except json.JSONDecodeError:
        res = {"raw_stdout": proc.stdout[-2000:]}
    return proc.returncode, wall, res, proc.stderr[-2000:], chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--export", help="print the verdicts in this run directory as item<TAB>key<TAB>verdict rows")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--skill", default="judge-mentions", choices=sorted(SKILLS))
    a = ap.parse_args()
    if a.export:
        for f in sorted(Path(a.export).glob("*.out.json")):
            for item, kv in json.loads(f.read_text()).items():
                print("".join(f"{item}\t{k}\t{v}\n" for k, v in kv.items()), end="")
        return
    if not a.out or not a.inputs:
        ap.error("--out and input chunks are required")
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    verdicts = SKILLS[a.skill]
    skill_hash = hashlib.sha256((HERE / f".claude/skills/{a.skill}/SKILL.md").read_bytes()).hexdigest()[:12]
    total_cost = 0.0
    for inp in a.inputs:
        stem = Path(inp).stem
        out = outdir / f"{stem}.out.json"
        chunk = json.loads(Path(inp).read_text())
        if not problems(chunk, out, verdicts):
            print(f"{stem}: done already, skipping", flush=True)
            continue
        for attempt in range(1 + a.retries):
            rc, wall, res, err, chunk = run_chunk(inp, out, a.model, a.timeout, a.skill)
            errs = problems(chunk, out, verdicts)
            meta = {"input": str(Path(inp).resolve()), "skill": a.skill, "model": a.model, "skill_sha": skill_hash,
                    "attempt": attempt, "returncode": rc, "wall_s": round(wall, 1),
                    "cost_usd": res.get("total_cost_usd"), "usage": res.get("usage"),
                    "num_turns": res.get("num_turns"), "result": str(res.get("result", ""))[:500],
                    "valid": not errs, "problems": errs[:20], "stderr": err}
            (outdir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=1))
            total_cost += res.get("total_cost_usd") or 0
            print(f"{stem}: attempt {attempt} rc={rc} {wall:.0f}s cost={res.get('total_cost_usd')} valid={not errs} {errs[:3]}", flush=True)
            if not errs:
                break
    print(f"total reported cost this run: {total_cost:.2f} USD", flush=True)


if __name__ == "__main__":
    sys.exit(main())
