"""Claude CLI judge backend; shared judging mechanics live in :mod:`judge.backend`."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

from .backend import (
    HERE,
    SPEC,
    SKILLS,
    JudgeBackend,
    expected_keys,
    export_rows,
    skill_sha,
    validate_response,
)


def problems(chunk, out_path, verdicts):
    """Compatibility helper for older scripts that validate a saved chunk output."""
    try:
        output = json.loads(Path(out_path).read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"unreadable: {exc}"]
    skill = next((name for name, labels in SKILLS.items() if labels == verdicts), None)
    return validate_response(chunk, output, skill) if skill else ["unknown verdict vocabulary"]


def claude_binary(binary=None):
    """Resolve the Claude executable from an argument, env var, or PATH."""
    binary = binary or os.environ.get("MIRAGE_CLAUDE_BIN") or shutil.which("claude")
    if not binary:
        raise FileNotFoundError("no claude binary: set MIRAGE_CLAUDE_BIN or put claude on PATH")
    return binary


def run_chunk(inp, out, model, timeout, skill, binary="claude"):
    """Preserve Mirage's existing Claude Code invocation for fidelity with frozen verdicts."""
    chunk = json.loads(Path(inp).read_text(encoding="utf-8"))
    if any(c.isspace() for c in f"{Path(inp).resolve()}{Path(out).resolve()}"):
        raise ValueError(f"the /{skill} invocation is space-separated; paths with whitespace break it: {inp}, {out}")
    dirs = {str(Path(inp).resolve().parent), str(Path(out).resolve().parent)}
    dirs |= {str(Path(entry["image"]).resolve().parent) for entry in chunk}
    cmd = [binary, "-p", f"/{skill} {Path(inp).resolve()} {Path(out).resolve()}",
           "--model", model, "--output-format", "json", "--allowedTools", "Read", "Write",
           "--setting-sources", "project", "--strict-mcp-config", "--no-session-persistence"]
    for directory in sorted(dirs):
        cmd += ["--add-dir", directory]
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return None, time.time() - start, {}, f"timed out after {timeout}s", chunk
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        result = {"raw_stdout": proc.stdout[-2000:]}
    return proc.returncode, time.time() - start, result, proc.stderr[-2000:], chunk


class ClaudeCLI(JudgeBackend):
    """Claude backend using the existing project skills and exact prior CLI options."""

    name = "claude"
    executable = "claude"
    binary_env = "MIRAGE_CLAUDE_BIN"

    def __init__(self, model=None, binary=None, workdir=None, chunk=10, timeout=1800, retries=1):
        super().__init__(model or SPEC["judge_model"], binary, workdir, chunk, timeout, retries)

    def invoke(self, inp: Path, out: Path, skill: str) -> Dict:
        out.unlink(missing_ok=True)
        rc, wall, result, stderr, _ = run_chunk(inp, out, self.model, self.timeout, skill, self.binary)
        return {
            "returncode": rc,
            "wall_s": round(wall, 1),
            "cost_usd": result.get("total_cost_usd"),
            "usage": result.get("usage"),
            "num_turns": result.get("num_turns"),
            "result": str(result.get("result", result.get("raw_stdout", "")))[:500],
            "stderr": stderr,
        }


def run_chunks(inputs, outdir, skill, model=None, timeout=1800, retries=1, binary="claude"):
    """Compatibility entry point backed by the shared batch runner; returns reported USD cost."""
    backend = ClaudeCLI(model=model, binary=binary, timeout=timeout, retries=retries)
    inputs = list(inputs)
    backend.run_files(inputs, outdir, skill)
    total = 0.0
    for inp in inputs:
        meta = json.loads((Path(outdir) / f"{Path(inp).stem}.meta.json").read_text(encoding="utf-8"))
        total += meta.get("cost_usd") or 0.0
    return total


def export(inputs: Iterable[Path], outdir: Path):
    return export_rows(inputs, outdir)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="*")
    parser.add_argument("--out")
    parser.add_argument("--export", help="export verdict rows from this run directory")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--skill", default="judge-mentions", choices=sorted(SKILLS))
    args = parser.parse_args()
    if not args.inputs:
        parser.error("input chunks are required")
    if args.export:
        sys.stdout.writelines(f"{item}\t{key}\t{verdict}\n" for item, key, verdict in export(args.inputs, args.export))
        return
    if not args.out:
        parser.error("--out is required")
    cost = run_chunks(args.inputs, args.out, args.skill, args.model, args.timeout, args.retries, claude_binary())
    print(f"total reported cost this run: {cost:.2f} USD", flush=True)


if __name__ == "__main__":
    main()
