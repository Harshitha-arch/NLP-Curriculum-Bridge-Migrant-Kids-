"""Load the preprocessing outputs used by the alignment stage."""

from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: Path | str) -> list[dict]:
    """All records of a jsonl file, in file order."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def content_only(records: list[dict]) -> list[dict]:
    """Records with ``section == "content"``, in their original order. Summaries and
    exercises repeat the teaching text and would add noise."""
    return [r for r in records if r["section"] == "content"]


def by_board(records: list[dict], board: str) -> list[dict]:
    """Records of one board, in their original order."""
    return [r for r in records if r["board"] == board]
