import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from app.ml_model import predecir_con_modelo
from app.schemas import PrediccionRequest
from app.template_reference import anexar_soporte_plantillas, puntajes_plantillas
from scripts.train_model import dividir_indices, seleccionar
from scripts.train_multilabel import metricas_multilabel, preparar_dataset


def y_pred_from_codes(classes: list[str], predictions: list[list[str]]) -> np.ndarray:
    index = {code: idx for idx, code in enumerate(classes)}
    matrix = np.zeros((len(predictions), len(classes)), dtype=int)
    for row_idx, codes in enumerate(predictions):
        for code in codes:
            idx = index.get(str(code))
            if idx is not None:
                matrix[row_idx, idx] = 1
    return matrix


def req_from_text(idx: int, text: str) -> PrediccionRequest:
    return PrediccionRequest(id_agenda=-(idx + 1), procedimiento_sistema='', hallazgos_conclusion=text, descripcion_estudio_013='')


def predecir_modelo(model_path: Path, texts: list[str]) -> tuple[list[list[str]], list[dict]]:
    predictions = []
    payloads = []
    for idx, text in enumerate(texts):
        payload, _ = predecir_con_modelo(req_from_text(idx, text), model_path)
        predictions.append([str(code) for code in payload.get('codigos') or []])
        payloads.append(payload)
    return predictions, payloads


def predecir_plantillas(texts: list[str]) -> tuple[list[list[str]], list[list[dict]]]:
    predictions = []
    rankings = []
    for text in texts:
        code_scores, matches = puntajes_plantillas(text)
        predictions.append(sorted(code_scores, key=code_scores.get, reverse=True))
        rankings.append(matches)
    return predictions, rankings


def predecir_con_soporte_visual(texts: list[str], model_payloads: list[dict]) -> tuple[list[list[str]], list[dict]]:
    predictions = []
    payloads = []
    for text, payload in zip(texts, model_payloads):
        enriched, _ = anexar_soporte_plantillas(text, payload, 0.0)
        predictions.append([str(code) for code in enriched.get('codigos') or []])
        payloads.append(enriched)
    return predictions, payloads


def resumen(nombre: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {'name': nombre, **metricas_multilabel(y_true, y_pred)}


def main() -> None:
    parser = argparse.ArgumentParser(description='Evalua modelo y soporte visual de plantillas en test 70/15/15.')
    parser.add_argument('--model', default='models/auditai_multilabel_tfidf_logreg.joblib')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--validation-size', type=float, default=0.15)
    parser.add_argument('--test-size', type=float, default=0.15)
    parser.add_argument('--sample-errors', type=int, default=10)
    args = parser.parse_args()

    texts, labels = preparar_dataset(args.limit)
    split_idx = dividir_indices(len(texts), args.validation_size, args.test_size)
    test_texts = seleccionar(texts, split_idx['test'])
    test_labels = seleccionar(labels, split_idx['test'])

    mlb = MultiLabelBinarizer()
    y_all = mlb.fit_transform(labels)
    y_test = y_all[split_idx['test']]
    classes = [str(code) for code in mlb.classes_]

    model_codes, model_payloads = predecir_modelo(Path(args.model), test_texts)
    template_codes, template_rankings = predecir_plantillas(test_texts)
    supported_codes, _ = predecir_con_soporte_visual(test_texts, model_payloads)

    results = [
        resumen('modelo_solo', y_test, y_pred_from_codes(classes, model_codes)),
        resumen('plantillas_solo_referencia', y_test, y_pred_from_codes(classes, template_codes)),
        resumen('modelo_con_soporte_visual', y_test, y_pred_from_codes(classes, supported_codes)),
    ]

    template_support = {
        'cases_with_template_match': sum(1 for matches in template_rankings if matches),
        'coverage': round(sum(1 for matches in template_rankings if matches) / len(template_rankings), 4) if template_rankings else 0.0,
        'avg_templates_per_case': round(sum(len(matches) for matches in template_rankings) / len(template_rankings), 4) if template_rankings else 0.0,
    }

    errors = []
    for idx, (actual, model_pred) in enumerate(zip(test_labels, model_codes)):
        actual_set = set(actual)
        model_set = set(model_pred)
        if model_set != actual_set and len(errors) < args.sample_errors:
            errors.append({
                'idx': idx,
                'actual': sorted(actual_set),
                'modelo': sorted(model_set),
                'missing_modelo': sorted(actual_set - model_set),
                'extra_modelo': sorted(model_set - actual_set),
                'plantillas_soporte': template_rankings[idx],
            })

    artifact = joblib.load(args.model)
    print(json.dumps({
        'model_path': args.model,
        'model': artifact.get('model') or artifact.get('engine'),
        'training_rows': len(texts),
        'test_size': len(test_texts),
        'unique_codes': len(classes),
        'results': results,
        'template_support': template_support,
        'sample_errors': errors,
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
