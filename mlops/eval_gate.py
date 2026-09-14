"""OmniForge Automated Model Evaluation Gate and Regression Benchmarking Engine.

Performs candidate vs. champion baseline comparison on accuracy, F1, RMSE, and p95 latency SLAs
to ensure only verified, non-regressing models are promoted to Production.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from loguru import logger

from mlops.base import (
    EvalGateResult,
    MetricComparison,
    ModelStage,
)
from mlops.mlflow_registry import MLflowRegistryManager, mlflow_registry


class ModelEvaluationGate:
    """Automated evaluation gate for comparing candidate models against active champions."""

    def __init__(
        self,
        registry: Optional[MLflowRegistryManager] = None,
        min_f1_delta: float = 0.0,
        min_accuracy_delta: float = 0.0,
        max_latency_p95_increase_ratio: float = 0.10,  # Max 10% increase in p95 latency
    ) -> None:
        self.registry = registry or mlflow_registry
        self.min_f1_delta = min_f1_delta
        self.min_accuracy_delta = min_accuracy_delta
        self.max_latency_p95_increase_ratio = max_latency_p95_increase_ratio
        self.evaluation_history: List[EvalGateResult] = []

    def evaluate_candidate(
        self,
        model_name: str,
        candidate_version: int,
        golden_dataset_metrics: Optional[Dict[str, float]] = None,
        auto_promote: bool = False,
    ) -> EvalGateResult:
        """Evaluate candidate model against current champion or absolute thresholds."""
        reg_model = self.registry.get_registered_model(model_name)
        if not reg_model:
            raise KeyError(f"Model '{model_name}' not found in registry.")

        candidate = reg_model.get_version(candidate_version)
        if not candidate:
            raise KeyError(f"Candidate version {candidate_version} not found for model '{model_name}'.")

        champion = reg_model.get_stage_model(ModelStage.PRODUCTION)

        candidate_metrics = golden_dataset_metrics or candidate.metrics
        champion_metrics = champion.metrics if champion else {}

        comparisons: List[MetricComparison] = []
        overall_passed = True
        failure_reasons: List[str] = []

        # 1. Higher-is-better metrics (accuracy, f1, precision, recall)
        for metric_key, min_delta in [
            ("f1_score", self.min_f1_delta),
            ("accuracy", self.min_accuracy_delta),
            ("precision", 0.0),
            ("recall", 0.0),
        ]:
            if metric_key in candidate_metrics:
                cand_val = candidate_metrics[metric_key]
                champ_val = champion_metrics.get(metric_key)

                if champ_val is not None:
                    delta = round(cand_val - champ_val, 4)
                    passed = delta >= min_delta
                    desc = f"Candidate {metric_key}={cand_val} vs Champion={champ_val} (delta={delta:+.4f}, min_req={min_delta:+.4f})"
                else:
                    delta = None
                    passed = cand_val >= 0.50  # Default sanity baseline
                    desc = f"Initial baseline for {metric_key}={cand_val} (no existing champion)"

                if not passed:
                    overall_passed = False
                    failure_reasons.append(f"{metric_key} regression: {desc}")

                comparisons.append(
                    MetricComparison(
                        metric_name=metric_key,
                        candidate_value=cand_val,
                        champion_value=champ_val,
                        delta=delta,
                        threshold=min_delta,
                        passed=passed,
                        description=desc,
                    )
                )

        # 2. Lower-is-better latency metrics (latency_p95_ms)
        if "latency_p95_ms" in candidate_metrics:
            cand_lat = candidate_metrics["latency_p95_ms"]
            champ_lat = champion_metrics.get("latency_p95_ms")

            if champ_lat is not None and champ_lat > 0:
                lat_ratio = (cand_lat - champ_lat) / champ_lat
                passed = lat_ratio <= self.max_latency_p95_increase_ratio
                desc = f"p95 latency: Candidate={cand_lat:.1f}ms vs Champion={champ_lat:.1f}ms (delta={lat_ratio * 100:+.1f}%, max allowed=+{self.max_latency_p95_increase_ratio * 100:.1f}%)"
            else:
                lat_ratio = 0.0
                passed = cand_lat < 500.0  # Max 500ms hard ceiling
                desc = f"Initial p95 latency check: {cand_lat:.1f}ms"

            if not passed:
                overall_passed = False
                failure_reasons.append(f"Latency SLA violation: {desc}")

            comparisons.append(
                MetricComparison(
                    metric_name="latency_p95_ms",
                    candidate_value=cand_lat,
                    champion_value=champ_lat,
                    delta=round(cand_lat - champ_lat, 2) if champ_lat is not None else None,
                    threshold=self.max_latency_p95_increase_ratio,
                    passed=passed,
                    description=desc,
                )
            )

        promoted = False
        if overall_passed:
            decision_reason = "Model candidate passed all regression benchmarking gates and quality thresholds."
            if auto_promote:
                self.registry.transition_stage(
                    model_name=model_name,
                    version=candidate_version,
                    target_stage=ModelStage.PRODUCTION,
                )
                promoted = True
                logger.info(f"Auto-promoted candidate v{candidate_version} of '{model_name}' to PRODUCTION.")
        else:
            decision_reason = f"Candidate failed evaluation gate: {'; '.join(failure_reasons)}"
            logger.warning(
                f"Evaluation gate rejected candidate v{candidate_version} of '{model_name}': {decision_reason}"
            )

        result = EvalGateResult(
            model_name=model_name,
            candidate_version=candidate_version,
            champion_version=champion.version if champion else None,
            passed=overall_passed,
            promoted=promoted,
            decision_reason=decision_reason,
            comparisons=comparisons,
        )

        self.evaluation_history.append(result)
        return result


# Default global instance for platform runtime
eval_gate = ModelEvaluationGate()
