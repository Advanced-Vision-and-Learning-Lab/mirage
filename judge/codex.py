"""Optional Codex CLI backend; shared judging mechanics live in :mod:`judge.backend`."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable

from .backend import (
    HERE,
    SKILLS,
    JudgeBackend,
    export_rows,
    judge_prompt,
    response_schema,
)

DEFAULT_MODEL = "gpt-6.1-sol"


def codex_binary(binary=None):
    binary = binary or os.environ.get("MIRAGE_CODEX_BIN") or shutil.which("codex")
    if not binary:
        raise FileNotFoundError("no Codex CLI: set MIRAGE_CODEX_BIN or put codex on PATH")
    return binary


def codex_version(binary: str) -> str:
    try:
        result = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return (result.stdout or result.stderr).strip() or f"exit-{result.returncode}"


def codex_prompt(chunk, skill):
    """Ask Codex to label Mirage-selected candidates, never extract its own candidates."""
    return judge_prompt(chunk, skill)


class CodexCLI(JudgeBackend):
    """Codex CLI backend using Mirage's existing Claude-authored rubric files."""

    name = "codex"
    executable = "codex"
    binary_env = "MIRAGE_CODEX_BIN"

    def __init__(self, model=None, binary=None, workdir=None, chunk=10, timeout=1800, retries=1):
        model = model or os.environ.get("MIRAGE_CODEX_JUDGE_MODEL") or DEFAULT_MODEL
        super().__init__(model, binary, workdir, chunk, timeout, retries)
        self.cli_version = codex_version(self.binary)

    def invoke(self, inp: Path, out: Path, skill: str) -> Dict:
        chunk = json.loads(inp.read_text(encoding="utf-8"))
        images = [str(Path(entry["image"]).resolve()) for entry in chunk]
        missing = [image for image in images if not Path(image).is_file()]
        if missing:
            raise FileNotFoundError(f"judge images are missing: {missing[:5]}")

        schema_path = out.with_suffix(".schema.json")
        prompt = codex_prompt(chunk, skill)
        schema_path.write_text(json.dumps(response_schema(chunk, skill), indent=2) + "\n", encoding="utf-8")
        out.unlink(missing_ok=True)
        cmd = [self.binary, "exec", "--model", self.model, "--sandbox", "read-only", "--ephemeral",
               "--output-schema", str(schema_path.resolve()), "--output-last-message", str(out.resolve())]
        if images:
            cmd.extend(("--image", *images))
        start = time.time()
        try:
            proc = subprocess.run(cmd, cwd=HERE.parent, input=prompt, capture_output=True, text=True,
                                  timeout=self.timeout, check=False)
            rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            rc, stdout, stderr = 124, "", f"timed out after {self.timeout}s"
        return {
            "returncode": rc,
            "wall_s": round(time.time() - start, 1),
            "codex_cli_version": self.cli_version,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "schema": str(schema_path.resolve()),
            "stdout": stdout[-4000:],
            "stderr": stderr[-4000:],
        }


def run_chunks(inputs: Iterable[Path], outdir: Path, skill: str, model=None, timeout=1800,
               retries=1, binary="codex"):
    backend = CodexCLI(model=model, binary=binary, timeout=timeout, retries=retries)
    backend.run_files(inputs, outdir, skill)


def export(inputs: Iterable[Path], outdir: Path):
    return export_rows(inputs, outdir)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--export", type=Path, help="export verdict rows from this run directory")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--skill", choices=sorted(SKILLS), default="judge-mentions")
    args = parser.parse_args()
    if not args.inputs:
        parser.error("input chunks are required")
    if args.export:
        sys.stdout.writelines(f"{item}\t{key}\t{verdict}\n"
                               for item, key, verdict in export(args.inputs, args.export))
        return
    if not args.out:
        parser.error("--out is required")
    backend = CodexCLI(model=args.model, timeout=args.timeout, retries=args.retries, binary=codex_binary())
    backend.run_files(args.inputs, args.out, args.skill)


if __name__ == "__main__":
    main()
