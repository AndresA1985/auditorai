# INICIO CAMBIO AUDITORAI TARIFARIO: métricas puras y alcance honesto del holdout.
"""No clinical contents, models, databases or third-party imports."""
from __future__ import annotations

from collections import Counter
import math
import random


def _divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def multilabel_metrics(y_true, y_pred, catalog=None) -> dict:
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Las particiones deben tener igual tamaño positivo.")
    true = [set(row) for row in y_true]
    predicted = [set(row) for row in y_pred]
    classes = sorted(set(catalog or []) | set().union(*true, *predicted))
    per_code = {}
    for code in classes:
        tp = sum(code in actual and code in proposed for actual, proposed in zip(true, predicted))
        fp = sum(code not in actual and code in proposed for actual, proposed in zip(true, predicted))
        fn = sum(code in actual and code not in proposed for actual, proposed in zip(true, predicted))
        precision, recall = _divide(tp, tp + fp), _divide(tp, tp + fn)
        per_code[code] = {"precision": precision, "recall": recall, "f1": _divide(2 * tp, 2 * tp + fp + fn), "support": tp + fn, "tp": tp, "fp": fp, "fn": fn}
    tp = sum(len(actual & proposed) for actual, proposed in zip(true, predicted))
    actual_count = sum(map(len, true))
    proposed_count = sum(map(len, predicted))
    support = sum(item["support"] for item in per_code.values())
    sample_dice = [_divide(2 * len(actual & proposed), len(actual) + len(proposed)) for actual, proposed in zip(true, predicted)]
    sample_jaccard = [_divide(len(actual & proposed), len(actual | proposed)) for actual, proposed in zip(true, predicted)]
    return {
        "precision_micro": _divide(tp, proposed_count), "recall_micro": _divide(tp, actual_count),
        "f1_micro": _divide(2 * tp, proposed_count + actual_count),
        "f1_macro": _divide(sum(item["f1"] for item in per_code.values()), len(classes)),
        "f1_weighted": _divide(sum(item["f1"] * item["support"] for item in per_code.values()), support),
        "avg_dice": sum(sample_dice) / len(true), "avg_code_overlap": sum(sample_jaccard) / len(true),
        "exact_code_group_accuracy": sum(actual == proposed for actual, proposed in zip(true, predicted)) / len(true),
        "abstention_rate": sum(not proposed for proposed in predicted) / len(true),
        "unsupported_prediction_rate": _divide(sum(len(proposed - actual) for actual, proposed in zip(true, predicted)), proposed_count),
        "per_code": per_code, "size": len(true), "zero_division": 0,
    }


def bootstrap_intervals(y_true, y_pred, *, iterations=300, seed=42, independent_groups=None) -> dict:
    """Bootstrap independent components, not correlated visits, when supplied."""
    if not y_true or len(y_true) != len(y_pred) or iterations < 2:
        raise ValueError("No hay muestras suficientes para intervalos.")
    groups = independent_groups or [[index] for index in range(len(y_true))]
    flattened = [index for group in groups for index in group]
    if sorted(flattened) != list(range(len(y_true))):
        raise ValueError("Los grupos deben cubrir una sola vez cada muestra.")
    rng, samples = random.Random(seed), {key: [] for key in ("f1_micro", "f1_macro", "exact_code_group_accuracy")}
    # INICIO CAMBIO AUDITORAI TARIFARIO: catálogo fijo incluye falsos positivos globales.
    catalog = sorted(
        set().union(*(set(row) for row in y_true))
        | set().union(*(set(row) for row in y_pred))
    )
    # FIN CAMBIO AUDITORAI TARIFARIO: catálogo fijo incluye falsos positivos globales.
    for _ in range(iterations):
        indices = [index for _ in range(len(groups)) for index in rng.choice(groups)]
        result = multilabel_metrics([y_true[index] for index in indices], [y_pred[index] for index in indices], catalog)
        for key in samples:
            samples[key].append(result[key])
    return {key: {"lower": sorted(values)[int(.025 * (iterations - 1))], "upper": sorted(values)[int(.975 * (iterations - 1))],
                  "level": .95, "iterations": iterations, "independent_groups": len(groups)} for key, values in samples.items()}


# INICIO CAMBIO AUDITORAI TARIFARIO: cohortes documentales puras compartidas por trainers.
def document_cohort_metrics(records, y_true, y_pred, indices) -> dict:
    if len(y_true) != len(y_pred) or len(y_true) != len(indices):
        raise ValueError("Las cohortes deben corresponder al mismo holdout.")
    groups = {
        "document_present": [],
        "document_missing": [],
        "document_empty": [],
        "ocr_failed_or_unreadable": [],
        "ocr_present": [],
        "document_label_conflict": [],
    }
    ocr_failures = {
        "ocr_no_disponible", "ocr_sin_codigos_verificables",
        "tiempo_ocr_agotado", "ocr_fallido",
    }
    for position, index in enumerate(indices):
        evidence = records[index].get("evidencia_tarifario") or {}
        codes = {item["codigo"] for item in evidence.get("codigos", [])}
        state = evidence.get("estado")
        warnings = set(evidence.get("advertencias") or [])
        if not evidence or state in {"sin_documento", "sin_archivo", "sin_pdf"}:
            groups["document_missing"].append(position)
        elif state == "sin_codigos":
            groups["document_empty"].append(position)
        elif state != "ilegible" and not warnings & ocr_failures:
            groups["document_present"].append(position)
        # Estado documental y resultado OCR son ejes no excluyentes: un PDF sin
        # códigos por timeout/fallo pertenece a ambas cohortes.
        if state == "ilegible" or warnings & ocr_failures:
            groups["ocr_failed_or_unreadable"].append(position)
        if codes and evidence.get("fuente") in {"ocr", "mixta"}:
            groups["ocr_present"].append(position)
        if codes and codes != set(records[index]["labels"]):
            groups["document_label_conflict"].append(position)
    return {
        name: {
            "size": len(positions),
            "metrics": multilabel_metrics(
                [y_true[position] for position in positions],
                [y_pred[position] for position in positions],
            ) if positions else None,
        }
        for name, positions in groups.items()
    }
# FIN CAMBIO AUDITORAI TARIFARIO: cohortes documentales puras compartidas por trainers.


def partition_metric_metadata(metrics: dict, size: int, dataset: str, training_scope="train_split"):
    if training_scope != "train_split":
        return None
    if dataset != "test_holdout" or isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("Solo se publican métricas del holdout positivo.")
    for key in ("f1_macro", "f1_weighted"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Métrica inválida.")
    return {**metrics, "size": size, "dataset": dataset}
# FIN CAMBIO AUDITORAI TARIFARIO: métricas puras y alcance honesto del holdout.
