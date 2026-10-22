"""Tests for file discovery in preprocessing.run."""

from pathlib import Path

import pytest

from preprocessing.run import discover


def touch(root: Path, rel: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_discover_flat_and_class_folder_layouts(tmp_path):
    for rel in [
        "NCERT/fecu101.pdf",  # flat: class from the NCERT code (f = Class 6)
        "NCERT/fecu1ps.pdf",  # prelims: excluded
        "SCERT/Term_2.pdf",  # flat, no code: default grade
        "NCERT/Class3/cemm101.pdf",  # folder layout
        "notes.pdf",  # no board folder: ignored
    ]:
        touch(tmp_path, rel)
    found = discover(tmp_path, exclude_stems="1ps$", default_grade=6)
    assert [(s.board, s.grade, s.stem, s.term) for s in found] == [
        ("NCERT", 3, "cemm101", None),
        ("NCERT", 6, "fecu101", None),
        ("SCERT", 6, "Term_2", 2),
    ]


def test_discover_needs_a_grade(tmp_path):
    touch(tmp_path, "SCERT/Term_1.pdf")
    with pytest.raises(ValueError, match="default_grade"):
        discover(tmp_path)
