"""Unit tests for MLflow Registry Manager and Experiment Tracking."""

import pytest

from mlops.base import ModelStage, PipelineStatus
from mlops.mlflow_registry import MLflowRegistryManager


@pytest.fixture
def registry():
    return MLflowRegistryManager()


def test_experiment_run_lifecycle(registry):
    run = registry.start_run(experiment_name="fraud_detection", tags={"env": "test"})
    assert run.experiment_name == "fraud_detection"
    assert run.tags["env"] == "test"
    assert run.status == PipelineStatus.SUCCESS

    registry.log_params({"n_estimators": 100, "max_depth": 10}, run_id=run.run_id)
    registry.log_metrics({"f1_score": 0.945, "accuracy": 0.952}, run_id=run.run_id)
    registry.log_artifact("s3://models/fraud_model.joblib", run_id=run.run_id)

    ended = registry.end_run(run_id=run.run_id, status=PipelineStatus.SUCCESS)
    assert ended.end_time is not None
    assert ended.parameters["n_estimators"] == 100
    assert ended.metrics["f1_score"] == 0.945
    assert len(ended.artifact_uris) == 1


def test_model_registration_and_versioning(registry):
    run1 = registry.start_run(experiment_name="credit_scoring")
    registry.log_metrics({"f1_score": 0.88}, run_id=run1.run_id)
    registry.end_run(run_id=run1.run_id)

    v1 = registry.register_model(
        name="credit_scorer",
        run_id=run1.run_id,
        description="Initial credit risk classifier",
    )
    assert v1.model_name == "credit_scorer"
    assert v1.version == 1
    assert v1.stage == ModelStage.NONE
    assert v1.metrics["f1_score"] == 0.88

    # Register version 2
    run2 = registry.start_run(experiment_name="credit_scoring")
    registry.log_metrics({"f1_score": 0.93}, run_id=run2.run_id)
    registry.end_run(run_id=run2.run_id)

    v2 = registry.register_model(
        name="credit_scorer",
        run_id=run2.run_id,
        description="Retrained with gradient boosting",
    )
    assert v2.version == 2

    model = registry.get_registered_model("credit_scorer")
    assert model is not None
    assert len(model.versions) == 2
    assert model.latest_version == 2


def test_stage_transition_and_automatic_archival(registry):
    run1 = registry.start_run(experiment_name="exp1")
    registry.end_run(run1.run_id)
    v1 = registry.register_model(name="nlp_classifier", run_id=run1.run_id)

    run2 = registry.start_run(experiment_name="exp1")
    registry.end_run(run2.run_id)
    v2 = registry.register_model(name="nlp_classifier", run_id=run2.run_id)

    # Transition v1 to PRODUCTION
    t1 = registry.transition_stage(
        model_name="nlp_classifier",
        version=1,
        target_stage=ModelStage.PRODUCTION,
    )
    assert t1.stage == ModelStage.PRODUCTION

    # Transition v2 to PRODUCTION (should automatically archive v1)
    t2 = registry.transition_stage(
        model_name="nlp_classifier",
        version=2,
        target_stage=ModelStage.PRODUCTION,
        archive_existing_versions=True,
    )
    assert t2.stage == ModelStage.PRODUCTION

    # Verify v1 was archived
    v1_updated = registry.get_registered_model("nlp_classifier").get_version(1)
    assert v1_updated.stage == ModelStage.ARCHIVED


def test_production_rollback(registry):
    run1 = registry.start_run()
    registry.end_run(run1.run_id)
    registry.register_model(name="churn_model", run_id=run1.run_id)

    run2 = registry.start_run()
    registry.end_run(run2.run_id)
    registry.register_model(name="churn_model", run_id=run2.run_id)

    # Promote v1 then v2
    registry.transition_stage("churn_model", 1, ModelStage.PRODUCTION)
    registry.transition_stage("churn_model", 2, ModelStage.PRODUCTION)

    # Roll back to v1
    rolled_back = registry.rollback_production("churn_model", fallback_version=1)
    assert rolled_back.version == 1
    assert rolled_back.stage == ModelStage.PRODUCTION

    # Verify v2 was demoted
    model = registry.get_registered_model("churn_model")
    assert model.get_version(2).stage == ModelStage.ARCHIVED
