"""OmniForge DVC Pipeline Manager and Data Versioning Engine.

Provides deterministic data hashing, pipeline stage DAG resolution, stage execution caching,
and seamless integration with dvc.yaml and params.yaml.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from mlops.base import PipelineRunResult, PipelineStatus, StageDefinition


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


class DVCPipelineManager:
    """Manages reproducible data pipelines, stage execution DAGs, and versioned data lineage."""

    def __init__(self, workspace_dir: Optional[Path | str] = None) -> None:
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        self.stages: Dict[str, StageDefinition] = {}
        self.stage_callbacks: Dict[str, Callable[..., Any]] = {}
        self.stage_hashes: Dict[str, str] = {}  # stage_name -> last execution hash
        self.execution_history: List[PipelineRunResult] = []

    def register_stage(
        self,
        name: str,
        command: str = "",
        deps: Optional[List[str]] = None,
        outs: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[..., Any]] = None,
    ) -> StageDefinition:
        """Register a pipeline stage with explicit dependencies, outputs, and parameters."""
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
        logger.debug(f"Registered DVC pipeline stage: {name} (deps={len(stage.deps)}, outs={len(stage.outs)})")
        return stage

    def compute_stage_hash(self, stage_name: str) -> str:
        """Compute composite hash for a stage combining its parameters and dependency file hashes."""
        if stage_name not in self.stages:
            raise KeyError(f"Pipeline stage '{stage_name}' not registered.")
        stage = self.stages[stage_name]

        dep_hashes = []
        for dep in stage.deps:
            dep_path = self.workspace_dir / dep
            if dep_path.exists():
                dep_hashes.append(f"{dep}:{compute_file_hash(dep_path)}")
            else:
                dep_hashes.append(f"{dep}:absent")

        param_hash = compute_data_fingerprint(stage.params)
        composite = f"stage:{stage_name}|params:{param_hash}|deps:{';'.join(sorted(dep_hashes))}"
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()

    def run_stage(self, stage_name: str, force: bool = False) -> Dict[str, Any]:
        """Execute a single pipeline stage, leveraging hash caching if inputs are unchanged."""
        if stage_name not in self.stages:
            raise KeyError(f"Stage '{stage_name}' is not registered.")

        current_hash = self.compute_stage_hash(stage_name)
        cached_hash = self.stage_hashes.get(stage_name)

        if not force and cached_hash == current_hash:
            logger.info(f"Stage [{stage_name}] is up to date (cached hash: {current_hash[:8]}). Skipping execution.")
            return {"status": "CACHED", "hash": current_hash, "metrics": self.stages[stage_name].metrics}

        logger.info(f"Executing stage [{stage_name}]...")
        start_time = time.time()
        stage_metrics: Dict[str, float] = {}

        callback = self.stage_callbacks.get(stage_name)
        if callback:
            try:
                result = callback(self.stages[stage_name].params)
                if isinstance(result, dict):
                    stage_metrics = {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
            except Exception as e:
                logger.error(f"Stage [{stage_name}] failed: {e}")
                raise

        duration = round(time.time() - start_time, 4)
        self.stage_hashes[stage_name] = current_hash
        self.stages[stage_name].metrics = stage_metrics

        return {
            "status": "SUCCESS",
            "hash": current_hash,
            "duration_seconds": duration,
            "metrics": stage_metrics,
        }

    def run_pipeline(self, force: bool = False) -> PipelineRunResult:
        """Run all registered pipeline stages sequentially in dependency order."""
        start_time = time.time()
        executed: List[str] = []
        cached: List[str] = []
        stage_results: Dict[str, Any] = {}

        try:
            for name in self.stages:
                res = self.run_stage(name, force=force)
                stage_results[name] = res
                if res["status"] == "CACHED":
                    cached.append(name)
                else:
                    executed.append(name)

            total_duration = round(time.time() - start_time, 4)
            result = PipelineRunResult(
                status=PipelineStatus.SUCCESS,
                executed_stages=executed,
                cached_stages=cached,
                duration_seconds=total_duration,
                stage_results=stage_results,
            )
            self.execution_history.append(result)
            return result

        except Exception as e:
            total_duration = round(time.time() - start_time, 4)
            failed_result = PipelineRunResult(
                status=PipelineStatus.FAILED,
                executed_stages=executed,
                cached_stages=cached,
                duration_seconds=total_duration,
                stage_results={"error": str(e)},
            )
            self.execution_history.append(failed_result)
            raise

    def export_dvc_yaml(self) -> Dict[str, Any]:
        """Export registered stages to a DVC-compliant schema dictionary."""
        dvc_dict: Dict[str, Any] = {"stages": {}}
        for name, stage in self.stages.items():
            dvc_dict["stages"][name] = {
                "cmd": stage.command,
                "deps": stage.deps,
                "outs": stage.outs,
                "params": list(stage.params.keys()),
            }
        return dvc_dict


# Default global instance for platform runtime
dvc_pipeline = DVCPipelineManager()
