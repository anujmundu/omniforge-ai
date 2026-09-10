"""Unit tests for Model Evaluation Gate and Regression Benchmarking."""

import pytest

from mlops.base import ModelStage
from mlops.eval_gate import ModelEvaluationGate
from mlops.mlflow_registry import MLflowRegistryManager


@pytest.fixture
def eval_setup():
    reg = MLflowRegistryManager()
    gate = ModelEvaluationGate(registry=reg, min_f1_delta=0.0, max_latency_p95_increase_ratio=0.10)

    # Register Champion (v1)
    run1 = reg.start_run(experiment_name="recsys")
    reg.log_metrics({"f1_score": 0.90, "accuracy": 0.91, "latency_p95_ms": 40.0}, run_id=run1.run_id)
    reg.end_run(run1.run_id)
    reg.register_model(name="recsys_model", run_id=run1.run_id)
    reg.transition_stage("recsys_model", 1, ModelStage.PRODUCTION)

    # Register Candidate 1 (v2: better F1, fast latency)
    run2 = reg.start_run(experiment_name="recsys")
    reg.log_metrics({"f1_score": 0.94, "accuracy": 0.95, "latency_p95_ms": 42.0}, run_id=run2.run_id)
    reg.end_run(run2.run_id)
    reg.register_model(name="recsys_model", run_id=run2.run_id)

    # Register Candidate 2 (v3: regressed F1)
    run3 = reg.start_run(experiment_name="recsys")
    reg.log_metrics({"f1_score": 0.85, "accuracy": 0.86, "latency_p95_ms": 38.0}, run_id=run3.run_id)
    reg.end_run(run3.run_id)
    reg.register_model(name="recsys_model", run_id=run3.run_id)

    # Register Candidate 3 (v4: great F1, but violates latency SLA +50%)
    run4 = reg.start_run(experiment_name="recsys")
    reg.log_metrics({"f1_score": 0.98, "accuracy": 0.98, "latency_p95_ms": 75.0}, run_id=run4.run_id)
    reg.end_run(run4.run_id)
    reg.register_model(name="recsys_model", run_id=run4.run_id)

    return gate, reg


def test_eval_gate_candidate_passes_and_autopromotes(eval_setup):
    gate, reg = eval_setup

    # Evaluate Candidate v2 (should pass and auto promote)
    res = gate.evaluate_candidate(
        model_name="recsys_model",
        candidate_version=2,
        auto_promote=True,
    )
    assert res.passed is True
    assert res.promoted is True
    assert res.champion_version == 1

    # Verify v2 is now in PRODUCTION and v1 is ARCHIVED
    model = reg.get_registered_model("recsys_model")
    assert model.get_version(2).stage == ModelStage.PRODUCTION
    assert model.get_version(1).stage == ModelStage.ARCHIVED


def test_eval_gate_candidate_fails_on_accuracy_regression(eval_setup):
    gate, reg = eval_setup

    # Evaluate Candidate v3 (regressed F1)
    res = gate.evaluate_candidate(
        model_name="recsys_model",
        candidate_version=3,
        auto_promote=True,
    )
    assert res.passed is False
    assert res.promoted is False
    assert "regression" in res.decision_reason


def test_eval_gate_candidate_fails_on_latency_violation(eval_setup):
    gate, reg = eval_setup

    # Evaluate Candidate v4 (exceeds +10% latency ratio)
    res = gate.evaluate_candidate(
        model_name="recsys_model",
        candidate_version=4,
        auto_promote=True,
    )
    assert res.passed is False
    assert res.promoted is False
    assert "Latency SLA violation" in res.decision_reason


def test_eval_gate_initial_baseline():
    reg = MLflowRegistryManager()
    gate = ModelEvaluationGate(registry=reg)

    run = reg.start_run()
    reg.log_metrics({"f1_score": 0.88, "latency_p95_ms": 50.0}, run_id=run.run_id)
    reg.end_run(run.run_id)
    reg.register_model(name="new_model", run_id=run.run_id)

    # Candidate 1 with no champion
    res = gate.evaluate_candidate(model_name="new_model", candidate_version=1)
    assert res.passed is True
    assert res.champion_version is None
