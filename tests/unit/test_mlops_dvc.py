"""Unit tests for DVC Pipeline Manager and Data Lineage Fingerprinting."""

import tempfile
from pathlib import Path

import pytest

from mlops.base import PipelineStatus
from mlops.dvc_pipeline import (
    DVCPipelineManager,
    compute_data_fingerprint,
    compute_file_hash,
)


@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        # Create some test dummy files
        (workspace / "data.csv").write_text("col1,col2\n1,2\n3,4", encoding="utf-8")
        (workspace / "features.csv").write_text("f1,f2\n0.1,0.2", encoding="utf-8")
        yield workspace


def test_file_hash_computation(temp_workspace):
    file_path = temp_workspace / "data.csv"
    h1 = compute_file_hash(file_path)
    assert isinstance(h1, str)
    assert len(h1) == 64

    # Identical content yields identical hash
    h2 = compute_file_hash(file_path)
    assert h1 == h2

    # Modified content yields different hash
    file_path.write_text("col1,col2\n100,200", encoding="utf-8")
    h3 = compute_file_hash(file_path)
    assert h1 != h3


def test_data_fingerprint():
    data1 = {"lr": 0.01, "epochs": 10}
    data2 = {"epochs": 10, "lr": 0.01}  # Key order difference
    assert compute_data_fingerprint(data1) == compute_data_fingerprint(data2)

    data3 = {"lr": 0.02, "epochs": 10}
    assert compute_data_fingerprint(data1) != compute_data_fingerprint(data3)


def test_pipeline_stage_registration(temp_workspace):
    manager = DVCPipelineManager(workspace_dir=temp_workspace)
    stage = manager.register_stage(
        name="preprocess",
        deps=["data.csv"],
        outs=["features.csv"],
        params={"strategy": "median"},
        callback=lambda p: {"features": 2},
    )
    assert stage.name == "preprocess"
    assert "preprocess" in manager.stages
    assert "data.csv" in stage.deps


def test_pipeline_execution_caching(temp_workspace):
    manager = DVCPipelineManager(workspace_dir=temp_workspace)
    call_count = 0

    def dummy_callback(params):
        nonlocal call_count
        call_count += 1
        return {"accuracy": 0.92}

    manager.register_stage(
        name="train",
        deps=["data.csv"],
        outs=["model.joblib"],
        params={"lr": 0.05},
        callback=dummy_callback,
    )

    # First run: should execute
    res1 = manager.run_stage("train")
    assert res1["status"] == "SUCCESS"
    assert call_count == 1
    assert res1["metrics"]["accuracy"] == 0.92

    # Second run without changes: should be CACHED
    res2 = manager.run_stage("train")
    assert res2["status"] == "CACHED"
    assert call_count == 1  # Callback not called again

    # Force run: should execute again
    res3 = manager.run_stage("train", force=True)
    assert res3["status"] == "SUCCESS"
    assert call_count == 2


def test_full_pipeline_run(temp_workspace):
    manager = DVCPipelineManager(workspace_dir=temp_workspace)
    manager.register_stage(
        name="step1",
        deps=["data.csv"],
        callback=lambda p: {"step1_metric": 10},
    )
    manager.register_stage(
        name="step2",
        deps=["features.csv"],
        callback=lambda p: {"step2_metric": 20},
    )

    result = manager.run_pipeline()
    assert result.status == PipelineStatus.SUCCESS
    assert len(result.executed_stages) == 2
    assert "step1" in result.executed_stages
    assert "step2" in result.executed_stages

    # Re-run pipeline: all should be cached
    result_cached = manager.run_pipeline()
    assert result_cached.status == PipelineStatus.SUCCESS
    assert len(result_cached.cached_stages) == 2


def test_export_dvc_yaml(temp_workspace):
    manager = DVCPipelineManager(workspace_dir=temp_workspace)
    manager.register_stage(
        name="train_stage",
        command="python train.py",
        deps=["data.csv"],
        outs=["model.joblib"],
        params={"batch_size": 32},
    )
    dvc_dict = manager.export_dvc_yaml()
    assert "stages" in dvc_dict
    assert "train_stage" in dvc_dict["stages"]
    assert dvc_dict["stages"]["train_stage"]["cmd"] == "python train.py"
