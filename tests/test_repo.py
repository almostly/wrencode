"""Rules the repository keeps itself to."""

from __future__ import annotations

import pathlib
import unittest


class TestRepoRules(unittest.TestCase):
    def test_no_noqa_markers(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        files = [
            *sorted((root / "src" / "wrencode").glob("*.py")),
            *sorted((root / "tests").glob("*.py")),
        ]
        offenders = [
            f"{f.name}:{n}"
            for f in files
            for n, line in enumerate(f.read_text().splitlines(), 1)
            if "noqa" in line and "test_no_noqa_markers" not in line
        ]
        self.assertEqual(offenders, [], "lint exceptions belong in pyproject.toml")
