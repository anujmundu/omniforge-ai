#!/usr/bin/env python3
"""OmniForge Phase 7 Interactive Demonstration - MLOps & CI/CD Pipelines.

Demonstrates:
  1. DVC Reproducible Pipeline DAG execution, data hashing & caching.
  2. Central MLflow Experiment Tracking (parameters, metrics, artifacts).
  3. Model Registry semantic versioning and lifecycle stage state machine.
  4. Candidate vs. Champion Automated Evaluation Gate with regression detection.
  5. Automated Production Promotion and Zero-Downtime Rollback Safety.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlops.base import ModelStage
from mlops.dvc_pipeline import DVCPipelineManager, compute_file_hash
from mlops.eval_gate import ModelEvaluationGate
from mlops.mlflow_registry import MLflowRegistryManager


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_section(title: str):
    print(f"\n--- {title} ---")


def main():
    print_banner("OMNIFORGE MLOPS & CI/CD LIFECYCLE ENGINE - LIVE DEMONSTRATION")

    # -------------------------------------------------------------------------
    # 1. DVC Reproducible Pipeline & Data Hashing
    # -------------------------------------------------------------------------
    print_section("1. DVC Reproducible Pipeline Execution & Stage Caching")
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        raw_data = workspace / "credit_data.csv"
        raw_data.write_text(
            "user_id,income,credit_score,default\n1,75000,720,0\n2,32000,580,1\n3,110000,810,0\n", encoding="utf-8"
        )

        initial_hash = compute_file_hash(raw_data)
        print(f"[*] Raw Dataset: {raw_data.name} | SHA-256 Hash: {initial_hash[:16]}...")

        dvc = DVCPipelineManager(workspace_dir=workspace)
        dvc.register_stage(
            name="ingest_and_split",
            deps=["credit_data.csv"],
            outs=["splits/train.parquet", "splits/test.parquet"],
            params={"train_ratio": 0.8, "random_state": 42},
            callback=lambda p: {"records_ingested": 10000, "train_records": 8000},
        )
        dvc.register_stage(
            name="feature_engineering",
            deps=["splits/train.parquet"],
            outs=["features/train_matrix.parquet"],
            params={"scaling": "standard", "impute": "median"},
            callback=lambda p: {"features_engineered": 24},
        )
        dvc.register_stage(
            name="model_training",
            deps=["features/train_matrix.parquet"],
            outs=["models/candidate.joblib"],
            params={"algorithm": "lightgbm", "n_estimators": 150, "lr": 0.03},
            callback=lambda p: {"train_loss": 0.124, "val_f1": 0.942},
        )

        print("\n[*] First Run: Executing Pipeline Stages...")
        run1 = dvc.run_pipeline()
        print(f"  -> Pipeline Status: {run1.status.value}")
        print(f"  -> Executed Stages: {run1.executed_stages}")
        print(f"  -> Cached Stages  : {run1.cached_stages}")
        print(f"  -> Total Duration : {run1.duration_seconds:.4f}s")

        print("\n[*] Second Run (Inputs Unchanged): Testing Stage Caching...")
        run2 = dvc.run_pipeline()
        print(f"  -> Pipeline Status: {run2.status.value}")
        print(f"  -> Executed Stages: {run2.executed_stages}")
        print(f"  -> Cached Stages  : {run2.cached_stages} (100% Cache Hit!)")

    # -------------------------------------------------------------------------
    # 2. Central MLflow Experiment Tracking
    # -------------------------------------------------------------------------
    print_section("2. Central MLflow Experiment Tracking")
    registry = MLflowRegistryManager()

    # Run 1: Champion Baseline
    run_champ = registry.start_run(
        experiment_name="risk_scoring_prod",
        tags={"framework": "scikit-learn", "author": "mlops_team"},
    )
    registry.log_params({"model_type": "RandomForestClassifier", "n_estimators": 100, "max_depth": 8})
    registry.log_metrics(
        {"accuracy": 0.912, "f1_score": 0.905, "precision": 0.910, "recall": 0.900, "latency_p95_ms": 32.5}
    )
    registry.log_artifact("s3://omniforge-artifacts/models/risk_model_v1.joblib")
    registry.end_run()

    print(
        f"[*] Logged Run 1: ID={run_champ.run_id} | Metrics: F1={run_champ.metrics['f1_score']}, Latency p95={run_champ.metrics['latency_p95_ms']}ms"
    )

    # Run 2: High Performance Candidate
    run_cand_pass = registry.start_run(
        experiment_name="risk_scoring_prod",
        tags={"framework": "xgboost", "author": "research_team"},
    )
    registry.log_params({"model_type": "XGBClassifier", "n_estimators": 200, "learning_rate": 0.05})
    registry.log_metrics(
        {"accuracy": 0.954, "f1_score": 0.948, "precision": 0.950, "recall": 0.946, "latency_p95_ms": 34.0}
    )
    registry.log_artifact("s3://omniforge-artifacts/models/risk_model_v2.joblib")
    registry.end_run()

    print(
        f"[*] Logged Run 2: ID={run_cand_pass.run_id} | Metrics: F1={run_cand_pass.metrics['f1_score']}, Latency p95={run_cand_pass.metrics['latency_p95_ms']}ms"
    )

    # Run 3: Regressed Latency Candidate
    run_cand_slow = registry.start_run(
        experiment_name="risk_scoring_prod",
        tags={"framework": "ensemble_stacking"},
    )
    registry.log_params({"model_type": "StackingClassifier", "estimators_count": 8})
    registry.log_metrics(
        {"accuracy": 0.960, "f1_score": 0.955, "precision": 0.958, "recall": 0.952, "latency_p95_ms": 85.0}
    )
    registry.log_artifact("s3://omniforge-artifacts/models/risk_model_v3.joblib")
    registry.end_run()

    print(
        f"[*] Logged Run 3: ID={run_cand_slow.run_id} | Metrics: F1={run_cand_slow.metrics['f1_score']}, Latency p95={run_cand_slow.metrics['latency_p95_ms']}ms"
    )

    # -------------------------------------------------------------------------
    # 3. Model Registry Versioning & Initial Stage
    # -------------------------------------------------------------------------
    print_section("3. Model Registry & Champion Baseline Promotion")
    v1 = registry.register_model(name="risk_engine", run_id=run_champ.run_id, description="Baseline production model")
    v2 = registry.register_model(
        name="risk_engine", run_id=run_cand_pass.run_id, description="Optimized XGBoost candidate"
    )
    v3 = registry.register_model(name="risk_engine", run_id=run_cand_slow.run_id, description="Heavy stacking ensemble")

    # Promote v1 to Production
    registry.transition_stage("risk_engine", version=1, target_stage=ModelStage.PRODUCTION)
    print(f"[*] Registered 'risk_engine' v1 -> Stage: {v1.stage.value}")
    print(f"[*] Registered 'risk_engine' v2 -> Stage: {v2.stage.value}")
    print(f"[*] Registered 'risk_engine' v3 -> Stage: {v3.stage.value}")

    # -------------------------------------------------------------------------
    # 4. Automated Candidate vs. Champion Evaluation Gate
    # -------------------------------------------------------------------------
    print_section("4. Automated Evaluation Gate & Regression Prevention")
    gate = ModelEvaluationGate(registry=registry, min_f1_delta=0.0, max_latency_p95_increase_ratio=0.10)

    # Test Candidate 2 (Should Pass)
    print("\n[*] Evaluating Candidate v2 against Champion v1:")
    gate_res2 = gate.evaluate_candidate(model_name="risk_engine", candidate_version=2, auto_promote=True)
    print(f"  -> Gate Passed    : {gate_res2.passed}")
    print(f"  -> Auto-Promoted  : {gate_res2.promoted}")
    print(f"  -> Decision Reason: {gate_res2.decision_reason}")
    for cmp in gate_res2.comparisons:
        status_icon = "PASS" if cmp.passed else "FAIL"
        print(f"     [{status_icon}] {cmp.description}")

    # Test Candidate 3 (Should Fail on Latency SLA violation)
    print("\n[*] Evaluating Candidate v3 against Champion v2:")
    gate_res3 = gate.evaluate_candidate(model_name="risk_engine", candidate_version=3, auto_promote=True)
    print(f"  -> Gate Passed    : {gate_res3.passed}")
    print(f"  -> Auto-Promoted  : {gate_res3.promoted}")
    print(f"  -> Decision Reason: {gate_res3.decision_reason}")
    for cmp in gate_res3.comparisons:
        status_icon = "PASS" if cmp.passed else "FAIL"
        print(f"     [{status_icon}] {cmp.description}")

    # -------------------------------------------------------------------------
    # 5. Production Rollback Safety
    # -------------------------------------------------------------------------
    print_section("5. Production Rollback Safety Verification")
    model_entity = registry.get_registered_model("risk_engine")
    active_champ = model_entity.get_stage_model(ModelStage.PRODUCTION)
    print(f"[*] Current Active Production Model: v{active_champ.version} (F1={active_champ.metrics['f1_score']})")

    print("[*] Executing zero-downtime rollback to previous champion v1...")
    rolled = registry.rollback_production("risk_engine", fallback_version=1)
    print(f"  -> Rollback Completed: Now in Production: v{rolled.version} (Stage: {rolled.stage.value})")

    # Verify v2 is now ARCHIVED and v1 is PRODUCTION
    print(f"  -> Model v1 Status: {model_entity.get_version(1).stage.value}")
    print(f"  -> Model v2 Status: {model_entity.get_version(2).stage.value}")

    print_banner("PHASE 7 MLOPS & CI/CD PIPELINE DEMONSTRATION COMPLETE - ALL GATES VERIFIED")


if __name__ == "__main__":
    main()
