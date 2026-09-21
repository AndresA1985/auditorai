# INICIO CAMBIO AUDITORAI TARIFARIO: Transformer prospectivo local, aislado y reproducible.
"""Fine-tune three tariff ablations without downloading or activating a model.

Importing this module is pure: torch/transformers/numpy/joblib are imported only
from the explicit CLI execution path.  Test doubles may exercise orchestration,
but their reports are marked as simulations and never produce model artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.features import FEATURE_SCHEMA_VERSION, build_features
from app.tariff_evaluation import (
    bootstrap_intervals,
    document_cohort_metrics,
    multilabel_metrics,
    partition_metric_metadata,
)
from scripts.tariff_snapshot_dataset import (
    SnapshotError,
    load_snapshot_dataset,
    resolve_split_manifest,
    shared_pipeline_metadata,
)

MODES = ("clinical", "document_only", "clinical_document")
EXCLUDED_CODES = {"93000", "99203", "99204", "99252"}
TRAINER_VERSION = "tariff_transformer_candidate_v1"
REAL_EXECUTION_KIND = "local_huggingface_finetune"
ESTABLISHED_TRANSFORMER_SOURCE = ROOT / "scripts" / "train_transformer_multilabel_base.py"
ESTABLISHED_BASELINE_ENCODER_ID = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(path: Path) -> str:
    """Hash relative names and bytes of an explicit local model/tokenizer tree."""
    if not path.is_dir():
        raise SnapshotError("modelo_o_tokenizer_local_inexistente")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise SnapshotError("modelo_o_tokenizer_local_vacio")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def established_transformer_contract(
    *, max_length: int, thresholds: list[float], selected_threshold=None
) -> dict:
    """Describe compatibility with the established Transformer training contract.

    The baseline encoder ID is provenance only. Runtime loading still requires the
    separately recorded explicit local encoder/tokenizer directories.
    """
    return {
        "source_path": "scripts/train_transformer_multilabel_base.py",
        "source_sha256": file_digest(ESTABLISHED_TRANSFORMER_SOURCE),
        "baseline_encoder_id": ESTABLISHED_BASELINE_ENCODER_ID,
        "baseline_encoder_id_is_reference_only": True,
        "architecture": {
            "model": "AutoModelForSequenceClassification",
            "tokenizer": "AutoTokenizer",
        },
        "loss": "BCEWithLogitsLoss",
        "max_length": max_length,
        "selection": {
            "metric": "f1_macro",
            "partition": "validation",
            "threshold_candidates": list(thresholds),
            "selected_threshold": selected_threshold,
            "test_access": "after_threshold_freeze",
            "min_labels": 0,
        },
        "loading_policy": "explicit_local_directories_only",
    }


def artifact_runtime_metadata(
    output_dir: Path,
    mode: str,
    *,
    max_length: int,
    thresholds: list[float],
    selected_threshold: float,
) -> dict:
    """Return runtime and portable paths, both anchored to the candidate run."""
    candidate_output = Path(output_dir).resolve()
    model_dir = (candidate_output / "models" / mode).resolve()
    active_models = (ROOT / "models").resolve()
    if candidate_output not in model_dir.parents or model_dir == active_models or active_models in model_dir.parents:
        raise SnapshotError("model_dir_debe_permanecer_en_salida_candidata")
    return {
        "model_dir": str(model_dir),
        "model_dir_relative": model_dir.relative_to(candidate_output).as_posix(),
        "established_transformer_compatibility": established_transformer_contract(
            max_length=max_length,
            thresholds=thresholds,
            selected_threshold=selected_threshold,
        ),
    }


def _binary_targets(labels: list[set[str]], classes: list[str]) -> list[list[float]]:
    return [[float(code in row) for code in classes] for row in labels]


def _validate_probabilities(values, rows: int, columns: int) -> list[list[float]]:
    matrix = [list(row) for row in values]
    if len(matrix) != rows or any(len(row) != columns for row in matrix):
        raise SnapshotError("dimensiones_de_probabilidad_invalidas")
    for row in matrix:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SnapshotError("probabilidad_invalida")
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise SnapshotError("probabilidad_invalida")
    return [[float(value) for value in row] for row in matrix]


def _prediction_sets(probabilities, classes, threshold, texts):
    return [
        {
            code
            for code, score in zip(classes, row)
            if score >= threshold and code not in EXCLUDED_CODES
        }
        if text.strip()
        else set()
        for row, text in zip(probabilities, texts)
    ]


def make_huggingface_backend(*, encoder_dir, tokenizer_dir, classes, options, work_dir):
    """Create the real backend from local files only; called only by explicit training."""
    import numpy as np
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    # INICIO CAMBIO AUDITORAI TARIFARIO: cada ablación fija RNG antes de abrir pesos.
    set_seed(int(options["seed"]))
    # FIN CAMBIO AUDITORAI TARIFARIO: cada ablación fija RNG antes de abrir pesos.

    class EncodedDataset(torch.utils.data.Dataset):
        def __init__(self, texts, targets, tokenizer, max_length):
            self.encodings = tokenizer(
                texts, truncation=True, padding=True, max_length=max_length
            )
            self.targets = targets

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
            item["labels"] = torch.tensor(self.targets[index], dtype=torch.float32)
            return item

    class LocalBackend:
        execution_kind = REAL_EXECUTION_KIND

        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_dir), local_files_only=True
            )
            id2label = {index: code for index, code in enumerate(classes)}
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(encoder_dir),
                local_files_only=True,
                num_labels=len(classes),
                problem_type="multi_label_classification",
                id2label=id2label,
                label2id={code: index for index, code in id2label.items()},
                ignore_mismatched_sizes=True,
            )
            self.trainer = None

        def fit(self, train_texts, train_targets, validation_texts, validation_targets):
            training = EncodedDataset(
                train_texts, train_targets, self.tokenizer, options["max_length"]
            )
            validation = EncodedDataset(
                validation_texts,
                validation_targets,
                self.tokenizer,
                options["max_length"],
            )
            arguments = TrainingArguments(
                output_dir=str(work_dir),
                num_train_epochs=options["epochs"],
                learning_rate=options["learning_rate"],
                per_device_train_batch_size=options["batch_size"],
                per_device_eval_batch_size=options["eval_batch_size"],
                weight_decay=options["weight_decay"],
                eval_strategy="epoch",
                save_strategy="no",
                logging_steps=options["logging_steps"],
                report_to=[],
                fp16=options["fp16"],
                seed=options["seed"],
                data_seed=options["seed"],
                disable_tqdm=True,
            )
            self.trainer = Trainer(
                model=self.model,
                args=arguments,
                train_dataset=training,
                eval_dataset=validation,
                tokenizer=self.tokenizer,
            )
            result = self.trainer.train()
            return {
                "training_performed": True,
                "backend": REAL_EXECUTION_KIND,
                "train_loss": float(result.training_loss),
            }

        def predict_probabilities(self, texts, *, partition):
            del partition
            targets = [[0.0] * len(classes) for _ in texts]
            dataset = EncodedDataset(texts, targets, self.tokenizer, options["max_length"])
            logits = self.trainer.predict(dataset).predictions
            return (1.0 / (1.0 + np.exp(-logits))).tolist()

        def save(self, destination):
            destination.mkdir(parents=True, exist_ok=False)
            self.trainer.save_model(str(destination))
            self.tokenizer.save_pretrained(str(destination))

    return LocalBackend()


def train_mode(
    records,
    split,
    mode,
    *,
    thresholds,
    dataset_kind,
    dataset_sha,
    split_sha,
    model_version,
    encoder_dir,
    tokenizer_dir,
    output_dir,
    options,
    backend_factory=make_huggingface_backend,
    bootstrap_iterations=300,
    evaluated_at=None,
):
    """Train/tune on train+validation and touch test only after threshold selection."""
    texts = [build_features(row, mode=mode) for row in records]
    labels = [set(row["labels"]) for row in records]
    effective_train = [index for index in split["train"] if texts[index].strip()]
    if not effective_train:
        return {"status": "not_trainable", "reason": "no_train_evidence", "mode": mode}, None
    classes = sorted(set().union(*(labels[index] for index in effective_train)) - EXCLUDED_CODES)
    if len(classes) < 2:
        return {"status": "not_trainable", "reason": "fewer_than_two_train_labels", "mode": mode}, None

    validation_indices = split["validation"]
    test_indices = split["test"]
    train_texts = [texts[index] for index in effective_train]
    validation_texts = [texts[index] for index in validation_indices]
    train_targets = _binary_targets([labels[index] for index in effective_train], classes)
    validation_targets = _binary_targets([labels[index] for index in validation_indices], classes)
    backend = backend_factory(
        encoder_dir=encoder_dir,
        tokenizer_dir=tokenizer_dir,
        classes=classes,
        options=options,
        work_dir=output_dir / "work" / mode,
    )
    receipt = backend.fit(train_texts, train_targets, validation_texts, validation_targets)
    execution_kind = getattr(backend, "execution_kind", "test_double_dispatch")
    real_training = bool(
        execution_kind == REAL_EXECUTION_KIND
        and isinstance(receipt, dict)
        and receipt.get("training_performed") is True
    )

    validation_probabilities = _validate_probabilities(
        backend.predict_probabilities(validation_texts, partition="validation"),
        len(validation_indices),
        len(classes),
    )
    validation_true = [labels[index] for index in validation_indices]
    experiments = []
    best = None
    for threshold in thresholds:
        predicted = _prediction_sets(
            validation_probabilities, classes, threshold, validation_texts
        )
        metrics = multilabel_metrics(validation_true, predicted)
        experiment = {"threshold": threshold, "validation_metrics": metrics}
        experiments.append(experiment)
        key = (metrics["f1_macro"], metrics["precision_micro"], threshold)
        if best is None or key > best[0]:
            best = (key, experiment)
    frozen_threshold = best[1]["threshold"]

    # The test partition is requested only after the validation-selected threshold is frozen.
    test_texts = [texts[index] for index in test_indices]
    test_probabilities = _validate_probabilities(
        backend.predict_probabilities(test_texts, partition="test"),
        len(test_indices),
        len(classes),
    )
    test_true = [labels[index] for index in test_indices]
    test_predicted = _prediction_sets(
        test_probabilities, classes, frozen_threshold, test_texts
    )
    test_metrics = partition_metric_metadata(
        multilabel_metrics(test_true, test_predicted), len(test_indices), "test_holdout"
    )
    positions = {index: position for position, index in enumerate(test_indices)}
    groups = [
        [positions[index] for index in component if index in positions]
        for component in split["components"]
    ]
    groups = [group for group in groups if group]
    intervals = bootstrap_intervals(
        test_true,
        test_predicted,
        iterations=bootstrap_iterations,
        independent_groups=groups,
    )
    # INICIO CAMBIO AUDITORAI TARIFARIO: mismas cohortes y contrafactual que TF-IDF.
    cohorts = document_cohort_metrics(
        records, test_true, test_predicted, test_indices
    )
    missing_texts = [
        build_features({**records[index], "evidencia_tarifario": None}, mode=mode)
        for index in test_indices
    ]
    missing_probabilities = _validate_probabilities(
        backend.predict_probabilities(
            missing_texts, partition="missing_document_counterfactual"
        ),
        len(test_indices),
        len(classes),
    )
    missing_predictions = _prediction_sets(
        missing_probabilities, classes, frozen_threshold, missing_texts
    )
    missing_counterfactual = multilabel_metrics(test_true, missing_predictions)
    # FIN CAMBIO AUDITORAI TARIFARIO: mismas cohortes y contrafactual que TF-IDF.
    timestamp = evaluated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    evaluation = {
        "evaluated": real_training,
        "execution_kind": execution_kind,
        "dataset_kind": dataset_kind,
        "dataset_sha256": dataset_sha,
        "split_sha256": split_sha,
        "selection_metric": "f1_macro_on_validation",
        "threshold_frozen_before_test": True,
        "validation_metrics": best[1]["validation_metrics"],
        "validation_experiments": experiments,
        "final_test_metrics" if real_training else "simulated_test_metrics": test_metrics,
        "confidence_intervals" if real_training else "simulated_confidence_intervals": intervals,
        "cohorts" if real_training else "simulated_cohorts": cohorts,
        "missing_document_counterfactual" if real_training else "simulated_missing_document_counterfactual": missing_counterfactual,
        "unseen_test_labels": sorted(set().union(*test_true) - set(classes)),
        "split_sizes": {name: len(split[name]) for name in ("train", "validation", "test")},
        "effective_train_size": len(effective_train),
        "clinical_performance_validated": False,
        "clinical_validation_gate": "pending_independent_operator_approval",
        "scope": (
            "test_double_dispatch_only"
            if not real_training
            else "synthetic_pipeline_only"
            if dataset_kind == "synthetic"
            else "reviewed_snapshot_holdout_unapproved"
        ),
    }
    report = {
        "status": "evaluated" if real_training else "simulated",
        "mode": mode,
        "training_performed": real_training,
        "best_params": {"threshold": frozen_threshold, "min_labels": 0},
        "evaluation": evaluation,
    }
    if not real_training:
        return report, None

    runtime_metadata = artifact_runtime_metadata(
        output_dir,
        mode,
        max_length=options["max_length"],
        thresholds=thresholds,
        selected_threshold=frozen_threshold,
    )
    model_dir = Path(runtime_metadata["model_dir"])
    backend.save(model_dir)
    artifact = {
        "version": 2,
        "model_version": model_version + "-" + mode,
        "trained_at": timestamp,
        "evaluated_at": timestamp,
        "model": "transformer_multilabel",
        "target": "reviewed_tariff_codes",
        "training_scope": "train_split",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_mode": mode,
        "allow_abstention": True,
        "dataset_kind": dataset_kind,
        "clinical_performance_validated": False,
        "clinical_validation_gate": "pending_independent_operator_approval",
        **runtime_metadata,
        "classes": classes,
        "threshold": frozen_threshold,
        "min_labels": 0,
        "max_length": options["max_length"],
        "training_rows": len(effective_train),
        "unique_codes": len(classes),
        "local_model_sha256": tree_digest(model_dir),
        "evaluation": evaluation,
    }
    return report, artifact


def _joblib_writer(value, path):
    import joblib

    joblib.dump(value, path)


def run_candidate(
    *,
    snapshots,
    output_dir,
    model_version,
    encoder_dir,
    tokenizer_dir,
    thresholds,
    options,
    allow_synthetic=False,
    split_manifest=None,
    backend_factory=make_huggingface_backend,
    artifact_writer=_joblib_writer,
    candidate_root=None,
    bootstrap_iterations=300,
):
    """Execute one isolated candidate run; injectable dependencies support offline tests."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,99}", model_version):
        raise SnapshotError("version_invalida")
    if not thresholds or any(isinstance(value, bool) or not 0 < value < 1 for value in thresholds):
        raise SnapshotError("umbrales_invalidos")
    source_encoder = Path(encoder_dir).resolve(strict=True)
    source_tokenizer = Path(tokenizer_dir).resolve(strict=True)
    encoder_sha = tree_digest(source_encoder)
    tokenizer_sha = tree_digest(source_tokenizer)
    allowed = Path(candidate_root or ROOT / "candidate_runs").resolve()
    target = Path(output_dir).resolve()
    if target == allowed or allowed not in target.parents or target.exists():
        raise SnapshotError("salida_debe_ser_nueva_dentro_de_candidate_runs")

    loaded = load_snapshot_dataset(snapshots)
    if loaded["data_kind"] == "synthetic" and not allow_synthetic:
        raise SnapshotError("synthetic_requires_explicit_flag")
    if not loaded["records"]:
        raise SnapshotError("no_eligible_predecision_snapshots")
    pipeline = shared_pipeline_metadata(ROOT)
    split = resolve_split_manifest(
        loaded["records"], loaded["manifest_sha256"], pipeline, split_manifest
    )
    target.mkdir(parents=True, mode=0o700, exist_ok=False)
    target.chmod(0o700)
    write_json(target / "split_manifest.json", split)
    split_sha = file_digest(target / "split_manifest.json")
    run_manifest = {
        "run_manifest_version": "tariff_candidate_run_v1",
        "trainer": "transformer",
        "trainer_version": TRAINER_VERSION,
        "trainer_source_sha256": pipeline["source_sha256"]["transformer_trainer"],
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_version": model_version,
        "dataset_kind": loaded["data_kind"],
        "dataset_sha256": loaded["manifest_sha256"],
        "split_sha256": split_sha,
        "local_sources": {
            "encoder_path": str(source_encoder),
            "encoder_sha256": encoder_sha,
            "tokenizer_path": str(source_tokenizer),
            "tokenizer_sha256": tokenizer_sha,
            "downloads_allowed": False,
        },
        "established_transformer_compatibility": established_transformer_contract(
            max_length=options["max_length"], thresholds=thresholds
        ),
        "arguments": {"modes": list(MODES), "thresholds": thresholds, **options},
        "environment": pipeline["environment"],
        "contains_clinical_data": False,
    }
    write_json(target / "run_manifest.json", run_manifest)
    reports, artifacts = [], []
    for mode in MODES:
        report, artifact = train_mode(
            loaded["records"],
            split,
            mode,
            thresholds=thresholds,
            dataset_kind=loaded["data_kind"],
            dataset_sha=loaded["manifest_sha256"],
            split_sha=split_sha,
            model_version=model_version,
            encoder_dir=source_encoder,
            tokenizer_dir=source_tokenizer,
            output_dir=target,
            options=options,
            backend_factory=backend_factory,
            bootstrap_iterations=bootstrap_iterations,
        )
        reports.append(report)
        if artifact is not None:
            path = target / (model_version + "-" + mode + ".joblib")
            artifact_writer(artifact, path)
            path.chmod(0o600)
            artifacts.append(
                {
                    "filename": path.name,
                    "sha256": file_digest(path),
                    "mode": mode,
                    "model_tree_sha256": artifact["local_model_sha256"],
                    "never_activated": True,
                }
            )
    final = {
        "model_version": model_version,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_kind": loaded["data_kind"],
        "clinical_improvement_established": False,
        "clinical_performance_validated": False,
        "clinical_validation_gate": "pending_independent_operator_approval",
        "automatic_promotion": False,
        "training_scope": "train_split",
        "dataset_sha256": loaded["manifest_sha256"],
        "split_sha256": split_sha,
        "loader_exclusions": loaded["excluded_counts"],
        "split_exclusions": split["excluded_counts"],
        "provenance_scope": loaded["provenance_scope"],
        "ablations": reports,
        "artifacts": artifacts,
        "fake_dispatch_is_training": False,
        "acceptance": "Independent clinical/operator approval remains mandatory; synthetic or simulated runs cannot qualify.",
    }
    write_json(target / "evaluation_report.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune three isolated tariff Transformer candidates from explicit local weights."
    )
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--encoder-model-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--split-manifest")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--thresholds", default="0.2,0.3,0.4,0.5")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    options = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_length": args.max_length,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "seed": args.seed,
        "fp16": bool(args.fp16),
    }
    result = run_candidate(
        snapshots=args.snapshots,
        output_dir=args.output_dir,
        model_version=args.model_version,
        encoder_dir=args.encoder_model_dir,
        tokenizer_dir=args.tokenizer_dir,
        thresholds=thresholds,
        options=options,
        allow_synthetic=args.allow_synthetic,
        split_manifest=args.split_manifest,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output_dir": str(Path(args.output_dir).resolve()),
                "artifact_count": len(result["artifacts"]),
                "automatic_promotion": False,
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (SnapshotError, ValueError, OSError, RuntimeError):
        print(
            json.dumps(
                {"ok": False, "reason": "dataset_model_or_parameters_invalid; inspect private metadata only"}
            )
        )
        raise SystemExit(2)
# FIN CAMBIO AUDITORAI TARIFARIO: Transformer prospectivo local, aislado y reproducible.
