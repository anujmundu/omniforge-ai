"""OmniForge DVC Pipeline Manager and Data Versioning Engine."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from loguru import logger
from mlops.base import StageDefinition

def compute_file_hash(file_path: Path | str) -> str:
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return hashlib.sha256(f"missing:{path}".encode()).hexdigest()
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_data_fingerprint(data: Any) -> str:
    if isinstance(data, (dict, list)):
        payload = json.dumps(data, sort_keys=True, default=str)
    else:
        payload = str(data)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

class DVCPipelineManager:
    def __init__(self, workspace_dir: Optional[Path | str] = None) -> None:
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        self.stages: Dict[str, StageDefinition] = {}
        self.stage_callbacks: Dict[str, Callable[..., Any]] = {}
        self.stage_hashes: Dict[str, str] = {}

    def register_stage(
        self,
        name: str,
        command: str = "",
        deps: Optional[List[str]] = None,
        outs: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[..., Any]] = None,
    ) -> StageDefinition:
        stage = StageDefinition(
            name=name,
            command=command or f"python -m scripts.{name}",
            deps=deps or [],
            outs=outs or [],
            params=params or {},
        )
        self.stages[name] = stage
        if callback:
            self.stage_callbacks[name] = callback
        return stage
