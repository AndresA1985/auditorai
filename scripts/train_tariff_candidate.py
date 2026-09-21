# INICIO CAMBIO AUDITORAI TARIFARIO: entrenamiento aislado con manifiesto prospectivo.
"""Train new artifacts only; never import app config/db or change the active model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.features import FEATURE_SCHEMA_VERSION, build_features
from app.tariff_evidence import _explicit_negation
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
TRAINER_VERSION = "tariff_tfidf_candidate_v1"


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _prediction_sets(probabilities, classes, threshold, texts):
    # No min_labels fallback: an empty evidence text cannot force a label.
    return [set(code for code, score in zip(classes, row) if score >= threshold and code not in EXCLUDED_CODES) if text.strip() else set()
            for row, text in zip(probabilities, texts)]


def train_mode(records, split, mode, *, c_values, thresholds, dataset_kind, dataset_sha, split_sha, model_version):
    # Imported only during explicit training with an operator-reviewed manifest.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    texts = [build_features(row, mode=mode) for row in records]
    labels = [set(row["labels"]) for row in records]
    train_indices = split["train"]
    effective_train = [index for index in train_indices if texts[index].strip()]
    if not effective_train:
        return {"status": "not_trainable", "reason": "no_train_evidence", "mode": mode}, None
    classes = sorted(set().union(*(labels[index] for index in effective_train)) - EXCLUDED_CODES)
    if len(classes) < 2:
        return {"status": "not_trainable", "reason": "fewer_than_two_train_labels", "mode": mode}, None
    mlb = MultiLabelBinarizer(classes=classes)
    y_train = mlb.fit_transform([labels[index] & set(classes) for index in effective_train])
    if any(y_train[:, column].sum() in {0, len(effective_train)} for column in range(len(classes))):
        return {"status": "not_trainable", "reason": "constant_label_in_train", "mode": mode}, None
    # Preserve the established TF-IDF settings from scripts/train_multilabel_base.py.
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, strip_accents="unicode", min_df=2, max_features=250000)
    x_train = vectorizer.fit_transform([texts[index] for index in effective_train])
    validation_indices = split["validation"]
    x_validation = vectorizer.transform([texts[index] for index in validation_indices])
    validation_true = [labels[index] for index in validation_indices]
    experiments, best, best_classifier = [], None, None
    convergence_failures = 0
    for c_value in c_values:
        classifier = OneVsRestClassifier(LogisticRegression(C=c_value, solver="liblinear", class_weight="balanced", max_iter=1000, random_state=42), n_jobs=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(x_train, y_train)
        if any(isinstance(item.message, ConvergenceWarning) for item in caught):
            convergence_failures += 1
            continue
        probabilities = classifier.predict_proba(x_validation)
        for threshold in thresholds:
            predicted = _prediction_sets(probabilities, classes, threshold, [texts[index] for index in validation_indices])
            metrics = multilabel_metrics(validation_true, predicted)
            item = {"C": c_value, "threshold": threshold, "validation_metrics": metrics}
            experiments.append(item)
            key = (metrics["f1_macro"], metrics["precision_micro"], threshold, -c_value)
            if best is None or key > best[0]:
                best, best_classifier = (key, item), classifier
    if best is None:
        return {"status": "not_trainable", "reason": "all_experiments_failed_convergence", "mode": mode}, None
    params = best[1]
    test_indices = split["test"]
    test_texts = [texts[index] for index in test_indices]
    test_true = [labels[index] for index in test_indices]
    # One evaluation of the frozen test set after selection on validation.
    test_probabilities = best_classifier.predict_proba(vectorizer.transform(test_texts))
    test_predicted = _prediction_sets(test_probabilities, classes, params["threshold"], test_texts)
    test_metrics = partition_metric_metadata(multilabel_metrics(test_true, test_predicted), len(test_indices), "test_holdout")
    position = {index: i for i, index in enumerate(test_indices)}
    components = [[position[index] for index in component if index in position] for component in split["components"]]
    components = [component for component in components if component]
    intervals = bootstrap_intervals(test_true, test_predicted, independent_groups=components)
    # A fixed missing-document counterfactual assesses degraded evidence; no tuning.
    missing_texts = [build_features({**records[index], "evidencia_tarifario": None}, mode=mode) for index in test_indices]
    missing_predictions = _prediction_sets(best_classifier.predict_proba(vectorizer.transform(missing_texts)), classes, params["threshold"], missing_texts)
    unseen = sorted(set().union(*test_true) - set(classes))
    review_count = 0
    for index, proposed in zip(test_indices, test_predicted):
        evidence = records[index].get("evidencia_tarifario") or {}
        doc_codes = {item["codigo"] for item in evidence.get("codigos", [])}
        absent = not evidence or (evidence.get("estado") in {"sin_documento", "sin_archivo", "sin_pdf"} and not doc_codes)
        contradiction = any(_explicit_negation(str(records[index].get("hallazgos_conclusion") or ""), reference)
                            for reference in evidence.get("codigos", []))
        if not proposed or (not absent and (not doc_codes or proposed != doc_codes or evidence.get("advertencias")
                or evidence.get("fuente") in {"ocr", "mixta"} or contradiction)):
            review_count += 1
    evaluated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    evaluation = {"evaluated": True, "dataset_kind": dataset_kind, "dataset_sha256": dataset_sha, "split_sha256": split_sha,
                  "established_tfidf": {"source": "scripts/train_multilabel_base.py", "analyzer": "char_wb", "ngram_range": [3, 5],
                      "strip_accents": "unicode", "lowercase": True, "min_df": 2, "max_features": 250000, "fit_scope": "train_split"},
                  "selection_metric": "f1_macro_on_validation", "split_sizes": {name: len(split[name]) for name in ("train", "validation", "test")},
                  "effective_train_size": len(effective_train), "validation_metrics": params["validation_metrics"], "final_test_metrics": test_metrics,
                  "confidence_intervals": intervals, "unseen_test_labels": unseen, "cohorts": document_cohort_metrics(records, test_true, test_predicted, test_indices),
                  "missing_document_counterfactual": multilabel_metrics(test_true, missing_predictions),
                  "documentary_review_rate": review_count / len(test_indices),
                  "review_rate_scope": "documentary discrepancy/OCR/negation plus abstention; excludes model citation verification and clinical review",
                  "convergence_failures": convergence_failures, "experiments": experiments,
                  "clinical_performance_validated": False,
                  "clinical_validation_gate": "pending_independent_operator_approval",
                  "scope": "synthetic_pipeline_only" if dataset_kind == "synthetic" else "reviewed_snapshot_holdout_unapproved"}
    artifact = {"version": 2, "model_version": model_version + "-" + mode, "trained_at": evaluated_at, "evaluated_at": evaluated_at,
                "model": "tfidf_logistic_regression_multilabel", "target": "reviewed_tariff_codes", "training_scope": "train_split",
                "feature_schema_version": FEATURE_SCHEMA_VERSION, "feature_mode": mode, "allow_abstention": True,
                "dataset_kind": dataset_kind, "clinical_performance_validated": False,
                "clinical_validation_gate": "pending_independent_operator_approval",
                "vectorizer": vectorizer, "classifier": best_classifier, "label_binarizer": mlb, "threshold": params["threshold"],
                "min_labels": 0, "training_rows": len(effective_train), "unique_codes": len(classes), "evaluation": evaluation}
    return {"status": "evaluated", "mode": mode, "best_params": {"C": params["C"], "threshold": params["threshold"], "min_labels": 0}, "evaluation": evaluation}, artifact


def main():
    parser = argparse.ArgumentParser(description="Train isolated tariff candidates from reviewed pre-decision snapshots; never activates a model.")
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--allow-synthetic", action="store_true", help="Test the pipeline only; never clinical performance.")
    parser.add_argument("--split-manifest", help="Reuse a frozen manifest from the SAME dataset hash.")
    parser.add_argument("--c-values", default="0.5,1.0,2.0")
    parser.add_argument("--thresholds", default="0.2,0.3,0.4,0.5")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,99}", args.model_version):
        raise SnapshotError("version_invalida")
    target = Path(args.output_dir).resolve()
    allowed = (ROOT / "candidate_runs").resolve()
    if target == allowed or allowed not in target.parents or target.exists():
        raise SnapshotError("salida_debe_ser_nueva_dentro_de_candidate_runs")
    loaded = load_snapshot_dataset(args.snapshots)
    if loaded["data_kind"] == "synthetic" and not args.allow_synthetic:
        raise SnapshotError("synthetic_requires_explicit_flag")
    records = loaded["records"]
    if not records:
        raise SnapshotError("no_eligible_predecision_snapshots")
    c_values = [float(item) for item in args.c_values.split(",")]
    thresholds = [float(item) for item in args.thresholds.split(",")]
    if not c_values or not thresholds or any(not 0 < value <= 100 for value in c_values) or any(not 0 < value < 1 for value in thresholds):
        raise SnapshotError("parametros_invalidos")
    pipeline_metadata = shared_pipeline_metadata(ROOT)
    split = resolve_split_manifest(
        records,
        loaded["manifest_sha256"],
        pipeline_metadata,
        args.split_manifest,
    )
    target.mkdir(parents=True, mode=0o700)
    target.chmod(0o700)
    write_json(target / "split_manifest.json", split)
    split_sha = digest(target / "split_manifest.json")
    # INICIO CAMBIO AUDITORAI TARIFARIO: argumentos/entorno quedan trazados sin contenido clínico.
    write_json(
        target / "run_manifest.json",
        {
            "run_manifest_version": "tariff_candidate_run_v1",
            "trainer": "tfidf",
            "trainer_version": TRAINER_VERSION,
            "trainer_source_sha256": pipeline_metadata["source_sha256"]["tfidf_trainer"],
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "model_version": args.model_version,
            "dataset_kind": loaded["data_kind"],
            "dataset_sha256": loaded["manifest_sha256"],
            "split_sha256": split_sha,
            "arguments": {
                "modes": list(MODES),
                "c_values": c_values,
                "thresholds": thresholds,
                "allow_synthetic": bool(args.allow_synthetic),
            },
            "environment": pipeline_metadata["environment"],
            "contains_clinical_data": False,
        },
    )
    # FIN CAMBIO AUDITORAI TARIFARIO: argumentos/entorno quedan trazados sin contenido clínico.
    import joblib
    reports, artifacts = [], []
    for mode in MODES:
        report, artifact = train_mode(records, split, mode, c_values=c_values, thresholds=thresholds,
                                      dataset_kind=loaded["data_kind"], dataset_sha=loaded["manifest_sha256"], split_sha=split_sha, model_version=args.model_version)
        reports.append(report)
        if artifact is not None:
            path = target / (args.model_version + "-" + mode + ".joblib")
            joblib.dump(artifact, path)
            path.chmod(0o600)
            artifacts.append({"filename": path.name, "sha256": digest(path), "mode": mode, "never_activated": True})
    final = {"model_version": args.model_version, "feature_schema_version": FEATURE_SCHEMA_VERSION, "dataset_kind": loaded["data_kind"],
             "clinical_improvement_established": False, "clinical_performance_validated": False,
             "clinical_validation_gate": "pending_independent_operator_approval",
             "automatic_promotion": False, "training_scope": "train_split",
             "dataset_sha256": loaded["manifest_sha256"], "loader_exclusions": loaded["excluded_counts"], "split_exclusions": split["excluded_counts"],
             "provenance_scope": loaded["provenance_scope"], "ablations": reports, "artifacts": artifacts,
             "acceptance": "Requires clinical reviewed snapshots, baseline comparison, rare-label/disagreement/OCR review and operator decision; synthetic runs cannot qualify."}
    write_json(target / "evaluation_report.json", final)
    print(json.dumps({"ok": True, "dataset_kind": loaded["data_kind"], "output_dir": str(target), "artifact_count": len(artifacts), "automatic_promotion": False}, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (SnapshotError, ValueError, OSError):
        # Paths and clinical row contents deliberately omitted from diagnostics.
        print(json.dumps({"ok": False, "reason": "dataset_or_parameters_invalid; inspect private metadata only"}))
        raise SystemExit(2)
# FIN CAMBIO AUDITORAI TARIFARIO: entrenamiento aislado con manifiesto prospectivo.
