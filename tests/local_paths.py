"""
Paths to data that only exists on one machine: private scans, model weights, another install.

They are read from tests/local_paths.yaml, which is not in git. To run the checks that use
them, copy tests/local_paths.example.yaml to tests/local_paths.yaml and fill in what this
machine has. A check whose path is not set skips that part and says so.
"""
from pathlib import Path

import yaml

FILE = Path(__file__).resolve().parent / "local_paths.yaml"


def local_path(key):
    """The path set for key in tests/local_paths.yaml; None without the file, the key or a value."""
    if not FILE.exists():
        return None
    value = (yaml.safe_load(FILE.read_text(encoding="utf-8")) or {}).get(key)
    return Path(value) if value else None


def not_set(key):
    """Why a part is skipped, for the output of a check."""
    return f"{key} is not set in tests/local_paths.yaml (see tests/local_paths.example.yaml)"
