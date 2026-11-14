"""Shared judge protocol: task schemas, rubric integrity, chunking, retries, and exports."""

import hashlib
import json
import inspect
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .store import MANIFEST

HERE = Path(__file__).resolve().parent
SPEC = json.loads(MANIFEST.read_text(encoding="utf-8"))

SKILLS = {
    "judge-mentions": {"present", "absent", "not_object", "unsure"},
    "judge-rope": {"correct", "related", "wrong", "unsure"},
    "judge-pope": {"present", "absent", "unsure"},
}


def expected_keys(chunk: List[Dict]) -> Dict[str, set]:
    """Return the per-image keys a grader is allowed and required to label."""
    if not chunk:
        return {}
    if "objects" in chunk[0]:
        return {entry["image_id"]: set(entry["objects"]) for entry in chunk}
    if "mentions" in chunk[0]:
        return {entry["image_id"]: {f'{m["model"]}|{m["word"]}' for m in entry["mentions"]} for entry in chunk}
    return {
        entry["image_id"]: {f'{box["box"]}|{answer["answer"]}'
                            for box in entry["boxes"] for answer in box["answers"]}
        for entry in chunk
    }


def skill_sha(skill: str) -> str:
    """Short digest of the Claude-authored rubric shared by every backend."""
    path = HERE / ".claude" / "skills" / skill / "SKILL.md"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def verify_skill(skill: str) -> str:
    """Return the rubric digest and refuse to mix labels from an untracked rubric change."""
    if skill not in SKILLS:
        raise ValueError(f"unknown judge skill: {skill}")
    digest = skill_sha(skill)
    frozen = SPEC["skills"][skill.removeprefix("judge-")]
    if digest != frozen:
        raise ValueError(f"{skill} rubric hash {digest} differs from the frozen {frozen}")
    return digest


def rubric_text(skill: str) -> str:
    """Return the rubric and verdict rules, omitting Claude-specific file invocation steps."""
    path = HERE / ".claude" / "skills" / skill / "SKILL.md"
    text = path.read_text(encoding="utf-8")
    if text.startswith("---\n"):
        _, _, text = text.partition("\n---\n")
    summary = text.split("## Invocation", 1)[0].strip()
    rules = "## Verdicts" + text.split("## Verdicts", 1)[1] if "## Verdicts" in text else ""
    return "\n\n".join(part for part in (summary, rules) if part)


def judge_prompt(chunk: List[Dict], skill: str) -> str:
    """Shared candidate-only prompt for backends that receive attached image inputs."""
    entries = []
    for index, entry in enumerate(chunk):
        row = {key: value for key, value in entry.items() if key != "image"}
        row["attached_image_index"] = index
        entries.append(row)
    return (
        "You are grading a fixed set of items for a vision-language-model benchmark.\n"
        "Judge only the words, answers, or objects explicitly listed in the input. Do not extract, "
        "add, remove, normalize, or rewrite candidates.\n"
        "The attached images are in the same order as the input entries; attached_image_index "
        "identifies the image for each entry. Inspect the corresponding image.\n"
        "Apply the rubric below. Return a verdict for every requested key, preserving exact image_id "
        "and key strings. Return only the JSON object required by the response schema.\n\n"
        f"Rubric ({skill}):\n{rubric_text(skill)}\n\n"
        f"Input entries:\n{json.dumps(entries, ensure_ascii=False, indent=2)}\n"
    )


def response_schema(chunk: List[Dict], skill: str) -> Dict:
    """Build a strict JSON schema enumerating every required key and allowed verdict."""
    labels = SKILLS[skill]
    properties = {}
    for item, keys in expected_keys(chunk).items():
        properties[item] = {
            "type": "object",
            "properties": {key: {"type": "string", "enum": sorted(labels)} for key in sorted(keys)},
            "required": sorted(keys),
            "additionalProperties": False,
        }
    return {"type": "object", "properties": properties, "required": sorted(properties),
            "additionalProperties": False}


def validate_response(chunk: List[Dict], output: Dict, skill: str) -> List[str]:
    """Check exact item/key coverage and verdict vocabulary for any backend response."""
    if not isinstance(output, dict):
        return ["output must be a JSON object"]
    labels, wanted, errors = SKILLS[skill], expected_keys(chunk), []
    for item, keys in wanted.items():
        got = output.get(item)
        if not isinstance(got, dict):
            errors.append(f"{item}: missing verdict object")
            continue
        missing, extra = keys - set(got), set(got) - keys
        bad = {key: value for key, value in got.items() if value not in labels}
        if missing:
            errors.append(f"{item}: {len(missing)} missing verdicts")
        if extra:
            errors.append(f"{item}: {len(extra)} unrequested verdicts")
        if bad:
            errors.append(f"{item}: invalid labels {bad}")
    extra_items = set(output) - set(wanted)
    if extra_items:
        errors.append(f"unexpected image ids: {sorted(extra_items)[:5]}")
    return errors


def output_problems(chunk: List[Dict], output_path: Path, skill: str) -> List[str]:
    try:
        output = json.loads(Path(output_path).read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"unreadable output: {exc}"]
    return validate_response(chunk, output, skill)


def export_rows(inputs: Iterable[Path], outdir: Path) -> List[Tuple[str, str, str]]:
    """Export only the requested ``(item, key, verdict)`` rows from a run directory."""
    rows = []
    for inp in inputs:
        chunk = json.loads(Path(inp).read_text(encoding="utf-8"))
        output = json.loads((Path(outdir) / f"{Path(inp).stem}.out.json").read_text(encoding="utf-8"))
        rows.extend((item, key, output[item][key])
                    for item, keys in expected_keys(chunk).items() for key in sorted(keys))
    return rows


def resolve_binary(binary: Optional[str], env_name: str, executable: str) -> str:
    binary = binary or os.environ.get(env_name) or shutil.which(executable)
    if not binary:
        raise FileNotFoundError(f"no {executable} binary: set {env_name} or put {executable} on PATH")
    return binary


class JudgeBackend:
    """Shared resumable batch runner. Subclasses implement only the backend invocation.

    Backends receive the same persisted input chunk and must write the same canonical JSON response.
    This keeps image selection, item keys, label validation, retries, and exports identical. ``invoke``
    is intentionally transport-neutral, so a future API backend can use this runner without changing
    the grading protocol.
    """

    name = "backend"
    executable = ""
    binary_env = ""

    def __init__(self, model: Optional[str], binary: Optional[str], workdir, chunk: int,
                 timeout: int, retries: int):
        self.model = model
        self.binary = resolve_binary(binary, self.binary_env, self.executable) if self.executable else binary
        self.workdir = Path(workdir) if workdir else None
        if chunk < 1:
            raise ValueError("chunk must be positive")
        self.chunk, self.timeout, self.retries = chunk, timeout, retries

    def revision_sha(self) -> str:
        """Fingerprint backend code so persistent outputs are not reused after prompt/runner edits."""
        sources = {inspect.getsourcefile(type(self)), inspect.getsourcefile(JudgeBackend)}
        source_paths = sorted(path for path in sources if path)
        if not source_paths:
            return "unknown"
        digest = hashlib.sha256()
        for source in source_paths:
            digest.update(Path(source).read_bytes())
        return digest.hexdigest()

    def invoke(self, inp: Path, out: Path, skill: str) -> Dict:
        """Run one backend session; return normalized process and usage metadata."""
        raise NotImplementedError

    def run_files(self, inputs: Iterable[Path], outdir: Path, skill: str) -> None:
        """Run prepared input files, reusing only complete outputs for the exact same run."""
        rubric_hash = verify_skill(skill)
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        for inp in inputs:
            inp = Path(inp)
            stem = inp.stem
            chunk = json.loads(inp.read_text(encoding="utf-8"))
            if not isinstance(chunk, list) or not chunk:
                raise ValueError(f"{inp}: expected a non-empty JSON list")
            output = outdir / f"{stem}.out.json"
            metadata = outdir / f"{stem}.meta.json"
            input_sha = hashlib.sha256(inp.read_bytes()).hexdigest()
            try:
                previous = json.loads(metadata.read_text(encoding="utf-8"))
            except Exception:
                previous = {}
            output_sha = hashlib.sha256(output.read_bytes()).hexdigest() if output.is_file() else None
            backend_sha = self.revision_sha()
            same_run = (previous.get("input_sha256") == input_sha and previous.get("rubric_sha") == rubric_hash
                        and previous.get("model") == self.model and previous.get("backend") == self.name
                        and previous.get("backend_sha256") == backend_sha
                        and previous.get("output_sha256") == output_sha and output_sha is not None
                        and previous.get("valid") is True)
            if same_run and not output_problems(chunk, output, skill):
                print(f"{stem}: done already, skipping", flush=True)
                continue

            for attempt in range(1 + self.retries):
                candidate = outdir / f"{stem}.attempt{attempt + 1}.out.json"
                candidate.unlink(missing_ok=True)
                result = self.invoke(inp, candidate, skill)
                try:
                    response = json.loads(candidate.read_text(encoding="utf-8"))
                    errors = validate_response(chunk, response, skill)
                except Exception as exc:
                    errors = [f"unreadable output: {exc}"]
                if result.get("returncode") not in (None, 0):
                    errors.insert(0, f"{self.name} exited with status {result['returncode']}")
                if not errors:
                    candidate.replace(output)
                meta = {
                    **result, "input": str(inp.resolve()), "input_sha256": input_sha,
                    "backend": self.name, "model": self.model, "skill": skill,
                    "rubric_sha": rubric_hash, "backend_sha256": backend_sha, "attempt": attempt + 1,
                    "valid": not errors, "problems": errors[:20],
                    "output": str(output.resolve()) if not errors else str(candidate.resolve()),
                }
                if not errors:
                    meta["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
                elif candidate.exists():
                    meta["output_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
                metadata.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
                print(f"{stem}: attempt {attempt + 1} rc={result.get('returncode')} "
                      f"{result.get('wall_s', 0):.0f}s valid={not errors} {errors[:3]}", flush=True)
                if not errors:
                    break
            else:
                raise RuntimeError(f"{skill}: {stem} failed validation after {1 + self.retries} attempts")

    def grade(self, kind: str, requests: List[Dict]) -> List[Tuple[str, str, str]]:
        """Grade Mirage-prepared requests and export their verdict rows."""
        skill = f"judge-{kind}"
        if skill not in SKILLS:
            raise ValueError(f"unsupported judge kind: {kind}")
        if not requests:
            return []
        work = self.workdir or Path(tempfile.mkdtemp(prefix=f"mirage-{self.name}-{skill}-"))
        work.mkdir(parents=True, exist_ok=True)
        inputs = []
        for offset in range(0, len(requests), self.chunk):
            inp = work / f"in_{offset // self.chunk:04d}.json"
            inp.write_text(json.dumps(requests[offset:offset + self.chunk], ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
            inputs.append(inp)
        try:
            self.run_files(inputs, work / "out", skill)
        except Exception as exc:
            exc.add_note(f"{skill} work directory (pass it as workdir to resume): {work}")
            raise
        return export_rows(inputs, work / "out")
