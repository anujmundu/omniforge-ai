"""OmniForge DVC Pipeline Manager and Data Versioning Engine."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any

def compute_file_hash(file_path: Path | str) -> str:
    """Compute deterministic SHA-256 hash for a file."""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return hashlib.sha256(f"missing:{path}".encode()).hexdigest()
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_data_fingerprint(data: Any) -> str:
    """Compute deterministic hash for arbitrary serializable data or strings."""
    if isinstance(data, (dict, list)):
        payload = json.dumps(data, sort_keys=True, default=str)
    else:
        payload = str(data)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
