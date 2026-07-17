"""Tests for dataset checksum utilities (E-6). Skips if the dataset isn't present (CI)."""
import os

import pytest
from src.data import download


def test_sha256_of_known_bytes(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    # sha256("hello")
    assert download.sha256(str(p)) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(download.__file__), "..", "..", "data", "SBAnational.csv")),
    reason="dataset not present (run `python -m src.data.download`)",
)
def test_dataset_matches_pinned_checksum():
    assert download.verify_checksum(strict=False)
