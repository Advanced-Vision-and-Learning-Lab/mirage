"""Shared judge protocol and backend parity tests; no live model calls."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from judge.backend import JudgeBackend, SKILLS, expected_keys, response_schema, validate_response
from judge.claude import ClaudeCLI
from judge.codex import CodexCLI


def _image(tmp_path):
    path = tmp_path / "image.jpg"
    path.write_bytes(b"test-image-bytes")
    return str(path)


def _mention_request(image):
    return {"image_id": "coco_1", "image": image,
            "mentions": [{"model": "model-a", "word": "cat", "context": "A cat is on the chair."}]}


class JudgeBackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_shared_schema_only_contains_mirage_candidates(self):
        chunk = [_mention_request(_image(self.root))]
        self.assertEqual(expected_keys(chunk), {"coco_1": {"model-a|cat"}})
        schema = response_schema(chunk, "judge-mentions")
        self.assertEqual(schema["properties"]["coco_1"]["properties"]["model-a|cat"]["enum"],
                         sorted(SKILLS["judge-mentions"]))
        self.assertEqual(validate_response(chunk, {"coco_1": {"model-a|cat": "present"}}, "judge-mentions"), [])
        self.assertTrue(validate_response(
            chunk, {"coco_1": {"model-a|cat": "present", "model-a|dog": "absent"}}, "judge-mentions"))

    def test_claude_uses_legacy_skill_invocation_through_shared_base(self):
        image = _image(self.root)
        captured = {}

        def fake_run(cmd, cwd, capture_output, text, timeout, check):
            captured.update(cmd=cmd, cwd=cwd, timeout=timeout)
            parts = cmd[2].split()
            Path(parts[2]).write_text(json.dumps({"coco_1": {"model-a|cat": "present"}}), encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout=json.dumps({"total_cost_usd": 0.0, "usage": {}}), stderr="")

        with patch("judge.claude.subprocess.run", fake_run):
            backend = ClaudeCLI(model="claude-test", binary="claude", workdir=self.root / "claude-run")
            rows = backend.grade("mentions", [_mention_request(image)])
        self.assertIsInstance(backend, JudgeBackend)
        self.assertEqual(rows, [("coco_1", "model-a|cat", "present")])
        self.assertEqual(captured["cwd"], ROOT / "judge")
        self.assertEqual(captured["cmd"][1], "-p")
        self.assertTrue(captured["cmd"][2].startswith("/judge-mentions "))
        self.assertEqual(captured["cmd"][captured["cmd"].index("--output-format") + 1], "json")
        self.assertEqual(captured["cmd"][captured["cmd"].index("--allowedTools") + 1:][:2], ["Read", "Write"])
        self.assertIn("--strict-mcp-config", captured["cmd"])

    def test_codex_backend_attaches_images_read_only_and_returns_only_requested_labels(self):
        image = _image(self.root)
        captured = {}

        def fake_run(cmd, **kwargs):
            if cmd == ["codex", "--version"]:
                return SimpleNamespace(returncode=0, stdout="codex-cli test", stderr="")
            captured.update(cmd=cmd, prompt=kwargs["input"])
            output = Path(cmd[cmd.index("--output-last-message") + 1])
            output.write_text(json.dumps({"coco_1": {"model-a|cat": "present"}}), encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("judge.codex.subprocess.run", fake_run):
            backend = CodexCLI(model="gpt-6.1-sol-test", binary="codex", workdir=self.root / "codex-run")
            rows = backend.grade("mentions", [_mention_request(image)])
        self.assertEqual(rows, [("coco_1", "model-a|cat", "present")])
        self.assertEqual(captured["cmd"][captured["cmd"].index("--sandbox") + 1], "read-only")
        self.assertEqual(captured["cmd"][captured["cmd"].index("--image") + 1], str(Path(image).resolve()))
        self.assertIn("Do not extract", captured["prompt"])

if __name__ == "__main__":
    unittest.main()
