import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from ml.anomaly.engine import AnomalyEngine
from ml.base import BaseMLEstimator, TaskType
from ml.classification.engine import ClassificationEngine
from ml.forecasting.engine import ForecastingEngine
from ml.regression.engine import RegressionEngine

DEFAULT_REGISTRY_DIR = Path("./storage/models")


class ModelRegistry:
    """Thread-safe model registry and in-memory cache for ultra-low-latency serving."""

    _instance: Optional["ModelRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls, storage_dir: Union[str, Path] = DEFAULT_REGISTRY_DIR):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.storage_dir = Path(storage_dir)
                cls._instance.storage_dir.mkdir(parents=True, exist_ok=True)
                cls._instance._cache: Dict[str, BaseMLEstimator] = {}
        return cls._instance

    @classmethod
    def get_engine_class(cls, task_type: Union[str, TaskType]) -> Type[BaseMLEstimator]:
        """Resolve engine class by task type."""
        task_str = task_type.value if isinstance(task_type, TaskType) else str(task_type).upper()
        if task_str == TaskType.CLASSIFICATION.value:
            return ClassificationEngine
        elif task_str == TaskType.REGRESSION.value:
            return RegressionEngine
        elif task_str == TaskType.ANOMALY_DETECTION.value:
            return AnomalyEngine
        elif task_str == TaskType.FORECASTING.value:
            return ForecastingEngine
        raise ValueError(f"Unknown task type: {task_type}")

    def register_and_save(self, model: BaseMLEstimator) -> str:
        """Persist model bundle and cache in memory."""
        saved_path = model.save(self.storage_dir)
        with self._lock:
            self._cache[model.model_id] = model
        return saved_path

    def get_model(self, model_id: str) -> BaseMLEstimator:
        """Retrieve model from cache or load from disk."""
        with self._lock:
            if model_id in self._cache:
                return self._cache[model_id]

        model_dir = self.storage_dir / model_id
        if not model_dir.exists():
            raise FileNotFoundError(f"Model '{model_id}' not found in registry at {self.storage_dir}")

        meta_path = model_dir / "metadata.json"
        task_type_str = None
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
                task_type_str = meta.get("task_type")

        # Fallback inspection by attempting to load
        for engine_cls in (ClassificationEngine, RegressionEngine, AnomalyEngine, ForecastingEngine):
            try:
                model = engine_cls.load(model_dir)
                with self._lock:
                    self._cache[model_id] = model
                return model
            except Exception:
                continue

        raise RuntimeError(f"Failed to deserialize model '{model_id}'.")

    def list_models(self) -> List[Dict[str, Any]]:
        """List all saved models and signatures."""
        models = []
        for p in self.storage_dir.iterdir():
            if p.is_dir() and (p / "model.joblib").exists():
                meta = {}
                meta_path = p / "metadata.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                    except Exception:
                        pass
                models.append(
                    {
                        "model_id": p.name,
                        "artifact_path": str(p / "model.joblib"),
                        "metadata": meta,
                    }
                )
        return models

    def clear_cache(self) -> None:
        """Flush the in-memory model cache."""
        with self._lock:
            self._cache.clear()


# Global singleton registry
registry = ModelRegistry()
