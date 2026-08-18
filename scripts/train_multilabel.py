import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from app.config import settings
from scripts.train_model import (
    cargar_dataset,
    dividir_indices,
    seleccionar,
    separar_codigos,
    texto_entrenamiento,
)


DEFAULT_C_VALUES = "0.5,1.0,2.0"
DEFAULT_THRESHOLDS = "0.20,0.30,0.40,0.50"
VALID_SELECTION_METRICS = {"f1_micro", "f1_samples", "avg_dice", "recall_micro", "precision_micro"}


def parse_float_list(value: str) -> list[float]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise argparse.ArgumentTypeError("Debe indicar al menos un valor numerico.")
    return values


def preparar_dataset(limit: int | None) -> tuple[list[str], list[tuple[str, ...]]]:
    rows = cargar_dataset(limit)
    texts = [texto_entrenamiento(row) for row in rows]
    labels = [separar_codigos(row.get("codigo_grupo_auditor") or "") for row in rows]
    pares = [(text, label) for text, label in zip(texts, labels) if text and label]
    if not pares:
        raise RuntimeError("No hay registros completos para entrenar.")
    return [text for text, _ in pares], [label for _, label in pares]


def construir_vectorizer(ngram_min: int, ngram_max: int, min_df: int, max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(ngram_min, ngram_max),
        strip_accents="unicode",
        lowercase=True,
        min_df=min_df,
        max_features=max_features,
    )


def construir_clasificador(c_value: float, max_iter: int, n_jobs: int, class_weight: str | None):
    estimator = LogisticRegression(
        C=c_value,
        solver="liblinear",
        class_weight=class_weight,
        max_iter=max_iter,
    )
    return OneVsRestClassifier(estimator, n_jobs=n_jobs)


def binarizar_con_threshold(probabilities: np.ndarray, threshold: float, min_labels: int) -> np.ndarray:
    pred = probabilities >= threshold
    if min_labels > 0:
        top_n = min(min_labels, probabilities.shape[1])
        for row_idx, row in enumerate(probabilities):
            if pred[row_idx].sum() < min_labels:
                top_idx = np.argpartition(row, -top_n)[-top_n:]
                pred[row_idx, top_idx] = True
    return pred.astype(int)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def metricas_multilabel(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    true_positive = np.logical_and(y_true == 1, y_pred == 1).sum(axis=1)
    predicted_positive = y_pred.sum(axis=1)
    actual_positive = y_true.sum(axis=1)

    total_tp = int(true_positive.sum())
    total_pred = int(predicted_positive.sum())
    total_actual = int(actual_positive.sum())

    precision_micro = safe_div(total_tp, total_pred)
    recall_micro = safe_div(total_tp, total_actual)
    f1_micro = safe_div(2 * precision_micro * recall_micro, precision_micro + recall_micro)

    precision_samples = np.mean([
        safe_div(tp, pred) for tp, pred in zip(true_positive, predicted_positive)
    ])
    recall_samples = np.mean([
        safe_div(tp, actual) for tp, actual in zip(true_positive, actual_positive)
    ])
    f1_samples = np.mean([
        safe_div(2 * tp, pred + actual)
        for tp, pred, actual in zip(true_positive, predicted_positive, actual_positive)
    ])
    exact_matches = np.all(y_true == y_pred, axis=1).mean()
    jaccard_samples = np.mean([
        safe_div(tp, pred + actual - tp)
        for tp, pred, actual in zip(true_positive, predicted_positive, actual_positive)
    ])

    return {
        "exact_code_group_accuracy": round(float(exact_matches), 4),
        "precision_micro": round(float(precision_micro), 4),
        "recall_micro": round(float(recall_micro), 4),
        "f1_micro": round(float(f1_micro), 4),
        "precision_samples": round(float(precision_samples), 4),
        "recall_samples": round(float(recall_samples), 4),
        "f1_samples": round(float(f1_samples), 4),
        "avg_dice": round(float(f1_samples), 4),
        "avg_code_overlap": round(float(jaccard_samples), 4),
        "avg_predicted_codes": round(float(predicted_positive.mean()), 4),
        "avg_actual_codes": round(float(actual_positive.mean()), 4),
    }


def evaluar_probabilidades(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    thresholds: Iterable[float],
    min_labels: int,
) -> list[dict]:
    results = []
    for threshold in thresholds:
        y_pred = binarizar_con_threshold(probabilities, threshold, min_labels)
        results.append({
            "threshold": threshold,
            "min_labels": min_labels,
            **metricas_multilabel(y_true, y_pred),
        })
    return results


def seleccionar_mejor(resultados: list[dict], metric: str) -> dict:
    if metric not in VALID_SELECTION_METRICS:
        raise RuntimeError("Metrica de seleccion no soportada: {0}".format(metric))
    return max(resultados, key=lambda item: (item[metric], item["f1_micro"], item["avg_dice"]))


def entrenar_y_evaluar(
    texts: list[str],
    labels: list[tuple[str, ...]],
    validation_size: float,
    test_size: float,
    c_values: list[float],
    thresholds: list[float],
    selection_metric: str,
    min_labels: int,
    ngram_min: int,
    ngram_max: int,
    min_df: int,
    max_features: int,
    max_iter: int,
    n_jobs: int,
    class_weight: str | None,
) -> dict:
    split_idx = dividir_indices(len(texts), validation_size, test_size)
    train_texts = seleccionar(texts, split_idx["train"])
    validation_texts = seleccionar(texts, split_idx["validation"])
    test_texts = seleccionar(texts, split_idx["test"])
    train_labels = seleccionar(labels, split_idx["train"])
    validation_labels = seleccionar(labels, split_idx["validation"])
    test_labels = seleccionar(labels, split_idx["test"])

    mlb = MultiLabelBinarizer()
    y_all = mlb.fit_transform(labels)
    y_train = y_all[split_idx["train"]]
    y_validation = y_all[split_idx["validation"]]
    y_test = y_all[split_idx["test"]]

    vectorizer = construir_vectorizer(ngram_min, ngram_max, min_df, max_features)
    x_train = vectorizer.fit_transform(train_texts)
    x_validation = vectorizer.transform(validation_texts)
    x_test = vectorizer.transform(test_texts)

    experiments = []
    best = None
    best_model = None

    for c_value in c_values:
        classifier = construir_clasificador(c_value, max_iter, n_jobs, class_weight)
        classifier.fit(x_train, y_train)
        validation_probabilities = classifier.predict_proba(x_validation)
        validation_results = evaluar_probabilidades(validation_probabilities, y_validation, thresholds, min_labels)
        validation_best = seleccionar_mejor(validation_results, selection_metric)
        experiment = {
            "params": {
                "C": c_value,
                "threshold": validation_best["threshold"],
                "min_labels": min_labels,
                "class_weight": class_weight or "none",
            },
            "validation_metrics": validation_best,
            "validation_grid": validation_results,
        }
        experiments.append(experiment)
        if best is None or validation_best[selection_metric] > best["validation_metrics"][selection_metric]:
            best = experiment
            best_model = classifier

    if best is None or best_model is None:
        raise RuntimeError("No se pudo entrenar ningun experimento.")

    threshold = float(best["params"]["threshold"])
    test_probabilities = best_model.predict_proba(x_test)
    test_pred = binarizar_con_threshold(test_probabilities, threshold, min_labels)
    final_test_metrics = {
        "name": "test",
        "size": len(test_texts),
        "threshold": threshold,
        "min_labels": min_labels,
        **metricas_multilabel(y_test, test_pred),
    }

    return {
        "vectorizer": vectorizer,
        "classifier": best_model,
        "label_binarizer": mlb,
        "evaluation": {
            "evaluated": True,
            "target": "codigo_grupo_auditor",
            "model": "tfidf_logistic_regression_multilabel",
            "selection_metric": selection_metric,
            "split": {
                "train_size": len(train_texts),
                "validation_size": len(validation_texts),
                "test_size": len(test_texts),
                "train_fraction": round(len(train_texts) / len(texts), 4),
                "validation_fraction": round(len(validation_texts) / len(texts), 4),
                "test_fraction": round(len(test_texts) / len(texts), 4),
            },
            "best_params": best["params"],
            "validation_metrics": best["validation_metrics"],
            "final_test_metrics": final_test_metrics,
            "experiments": experiments,
        },
    }


def entrenar_final(texts: list[str], labels: list[tuple[str, ...]], best_params: dict, args: argparse.Namespace):
    mlb = MultiLabelBinarizer()
    y_all = mlb.fit_transform(labels)
    vectorizer = construir_vectorizer(args.ngram_min, args.ngram_max, args.min_df, args.max_features)
    x_all = vectorizer.fit_transform(texts)
    classifier = construir_clasificador(
        float(best_params["C"]),
        args.max_iter,
        args.n_jobs,
        None if best_params["class_weight"] == "none" else best_params["class_weight"],
    )
    classifier.fit(x_all, y_all)
    return vectorizer, classifier, mlb


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena TF-IDF + LogisticRegression multi-label para codigo_grupo_auditor.")
    parser.add_argument("--output", default="models/auditai_multilabel_tfidf_logreg.joblib")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validation-size", type=float, default=settings.validation_size)
    parser.add_argument("--test-size", type=float, default=settings.test_size)
    parser.add_argument("--c-values", type=parse_float_list, default=parse_float_list(DEFAULT_C_VALUES))
    parser.add_argument("--thresholds", type=parse_float_list, default=parse_float_list(DEFAULT_THRESHOLDS))
    parser.add_argument("--selection-metric", choices=sorted(VALID_SELECTION_METRICS), default="avg_dice")
    parser.add_argument("--min-labels", type=int, default=1)
    parser.add_argument("--ngram-min", type=int, default=3)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=250000)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--final-fit-all", action="store_true", help="Reentrena el artefacto final con todo el dataset usando los mejores parametros.")
    args = parser.parse_args()

    class_weight = None if args.class_weight == "none" else args.class_weight
    texts, labels = preparar_dataset(args.limit)
    result = entrenar_y_evaluar(
        texts,
        labels,
        args.validation_size,
        args.test_size,
        args.c_values,
        args.thresholds,
        args.selection_metric,
        args.min_labels,
        args.ngram_min,
        args.ngram_max,
        args.min_df,
        args.max_features,
        args.max_iter,
        args.n_jobs,
        class_weight,
    )

    vectorizer = result["vectorizer"]
    classifier = result["classifier"]
    label_binarizer = result["label_binarizer"]
    training_scope = "train_split"
    if args.final_fit_all:
        vectorizer, classifier, label_binarizer = entrenar_final(texts, labels, result["evaluation"]["best_params"], args)
        training_scope = "all_rows_after_evaluation"

    artifact = {
        "version": 1,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model": "tfidf_logistic_regression_multilabel",
        "target": "codigo_grupo_auditor",
        "training_scope": training_scope,
        "vectorizer": vectorizer,
        "classifier": classifier,
        "label_binarizer": label_binarizer,
        "threshold": result["evaluation"]["best_params"]["threshold"],
        "min_labels": args.min_labels,
        "training_rows": len(texts),
        "unique_codes": len(label_binarizer.classes_),
        "evaluation": result["evaluation"],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)

    print(json.dumps({
        "output": str(output),
        "model": artifact["model"],
        "target": artifact["target"],
        "training_scope": artifact["training_scope"],
        "training_rows": artifact["training_rows"],
        "unique_codes": artifact["unique_codes"],
        "evaluation": artifact["evaluation"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
