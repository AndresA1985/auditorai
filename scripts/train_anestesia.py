import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error

from app.anestesia_model import normalizar_codigo, normalizar_tiempo, texto_codigo
from app.db import fetch_all
from scripts.train_model import dividir_indices, seleccionar

VALID_ENGINES = {'code_majority', 'tfidf_logreg'}
DEFAULT_ALLOWED_TIMES = '1,2,3,4,5,6,7,8,24'


def parse_values(value: str) -> set[str]:
    return {item.strip() for item in value.split(',') if item.strip()}


def numeric_sort(values):
    return sorted(values, key=lambda item: float(item))


def cargar_dataset(limit: int | None, allowed_times: set[str]) -> list[dict]:
    sql = '''
        SELECT
            id_agenda,
            fecha,
            id_empresa,
            id_seguro,
            codigo,
            tiempo_anestesia,
            procedimiento_auditor,
            nombre_procedimiento,
            hallazgo,
            conclusion
        FROM ap_auditoria_doctor_detalle
        WHERE estado = 1
          AND codigo IS NOT NULL
          AND codigo <> ''
          AND tiempo_anestesia IS NOT NULL
          AND tiempo_anestesia <> ''
        ORDER BY fecha ASC, id_agenda ASC, codigo ASC
    '''
    if limit:
        sql += '\nLIMIT {0}'.format(int(limit))
    rows = []
    for row in fetch_all(sql):
        code = normalizar_codigo(row.get('codigo'))
        tiempo = normalizar_tiempo(row.get('tiempo_anestesia'))
        if not code or tiempo not in allowed_times:
            continue
        row['codigo_norm'] = code
        row['tiempo_norm'] = tiempo
        rows.append(row)
    if not rows:
        raise RuntimeError('No hay registros de tiempo_anestesia con valores permitidos.')
    return rows


def texto_row(row: dict) -> str:
    text = ' '.join([
        str(row.get('procedimiento_auditor') or ''),
        str(row.get('nombre_procedimiento') or ''),
        str(row.get('hallazgo') or ''),
        str(row.get('conclusion') or ''),
    ]).strip()
    return texto_codigo(text, row['codigo_norm'])


def majority(values: list[str]) -> str:
    return Counter(values).most_common(1)[0][0]


def entrenar_mayoria(rows: list[dict], train_idx: list[int]) -> dict:
    by_code_values = defaultdict(list)
    global_values = []
    for row in seleccionar(rows, train_idx):
        by_code_values[row['codigo_norm']].append(row['tiempo_norm'])
        global_values.append(row['tiempo_norm'])
    return {
        'engine': 'code_majority',
        'by_code': {code: majority(values) for code, values in by_code_values.items()},
        'global_default': majority(global_values),
        'vectorizer': None,
        'classifier': None,
    }


def entrenar_tfidf(rows: list[dict], train_idx: list[int], max_features: int, class_weight: str | None) -> dict:
    train_rows = seleccionar(rows, train_idx)
    texts = [texto_row(row) for row in train_rows]
    y_train = [row['tiempo_norm'] for row in train_rows]
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        strip_accents='unicode',
        lowercase=True,
        min_df=2,
        max_features=max_features,
    )
    x_train = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(
        C=1.0,
        solver='liblinear',
        class_weight=class_weight,
        max_iter=1000,
    )
    classifier.fit(x_train, y_train)
    majority_artifact = entrenar_mayoria(rows, train_idx)
    return {
        'engine': 'tfidf_logreg',
        'by_code': majority_artifact['by_code'],
        'global_default': majority_artifact['global_default'],
        'vectorizer': vectorizer,
        'classifier': classifier,
    }


def predecir_artifact(artifact: dict, rows: list[dict]) -> list[str]:
    if artifact['engine'] == 'tfidf_logreg':
        x = artifact['vectorizer'].transform([texto_row(row) for row in rows])
        return [normalizar_tiempo(value) for value in artifact['classifier'].predict(x)]
    return [normalizar_tiempo(artifact['by_code'].get(row['codigo_norm']) or artifact['global_default']) for row in rows]


def evaluar(nombre: str, artifact: dict, rows: list[dict]) -> dict:
    y_true = [row['tiempo_norm'] for row in rows]
    y_pred = predecir_artifact(artifact, rows)
    labels = numeric_sort(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    y_true_num = [float(value) for value in y_true]
    y_pred_num = [float(value) for value in y_pred]
    return {
        'name': nombre,
        'size': len(rows),
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'macro_f1': round(float(report['macro avg']['f1-score']), 4),
        'weighted_f1': round(float(report['weighted avg']['f1-score']), 4),
        'mae_hours': round(float(mean_absolute_error(y_true_num, y_pred_num)), 4),
        'labels': labels,
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        'classification_report': report,
    }


def entrenar_engine(engine: str, rows: list[dict], train_idx: list[int], max_features: int, class_weight: str | None) -> dict:
    if engine == 'tfidf_logreg':
        return entrenar_tfidf(rows, train_idx, max_features, class_weight)
    return entrenar_mayoria(rows, train_idx)


def main() -> None:
    parser = argparse.ArgumentParser(description='Entrena modelo separado para tiempo de anestesia por codigo.')
    parser.add_argument('--output', default='models/auditai_anestesia.joblib')
    parser.add_argument('--engine', choices=sorted(VALID_ENGINES), default='code_majority')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--validation-size', type=float, default=0.15)
    parser.add_argument('--test-size', type=float, default=0.15)
    parser.add_argument('--allowed-times', default=DEFAULT_ALLOWED_TIMES)
    parser.add_argument('--max-features', type=int, default=100000)
    parser.add_argument('--class-weight', choices=['balanced', 'none'], default='balanced')
    parser.add_argument('--final-fit-all', action='store_true')
    args = parser.parse_args()

    allowed = parse_values(args.allowed_times)
    rows = cargar_dataset(args.limit, allowed)
    split = dividir_indices(len(rows), args.validation_size, args.test_size)
    class_weight = None if args.class_weight == 'none' else args.class_weight

    artifact = entrenar_engine(args.engine, rows, split['train'], args.max_features, class_weight)
    validation_rows = seleccionar(rows, split['validation'])
    test_rows = seleccionar(rows, split['test'])
    evaluation = {
        'evaluated': True,
        'target': 'tiempo_anestesia_por_codigo',
        'split': {
            'train_size': len(split['train']),
            'validation_size': len(split['validation']),
            'test_size': len(split['test']),
            'train_fraction': round(len(split['train']) / len(rows), 4),
            'validation_fraction': round(len(split['validation']) / len(rows), 4),
            'test_fraction': round(len(split['test']) / len(rows), 4),
        },
        'validation_metrics': evaluar('validation', artifact, validation_rows),
        'final_test_metrics': evaluar('test', artifact, test_rows),
    }

    training_scope = 'train_split'
    if args.final_fit_all:
        artifact = entrenar_engine(args.engine, rows, list(range(len(rows))), args.max_features, class_weight)
        training_scope = 'all_rows_after_evaluation'

    artifact.update({
        'version': 1,
        'trained_at': datetime.now().isoformat(timespec='seconds'),
        'model': 'anestesia_por_codigo',
        'target': 'tiempo_anestesia_por_codigo',
        'training_scope': training_scope,
        'training_rows': len(rows),
        'unique_codes': len({row['codigo_norm'] for row in rows}),
        'allowed_times': numeric_sort(allowed),
        'evaluation': evaluation,
    })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)

    print(json.dumps({
        'output': str(output),
        'model': artifact['model'],
        'engine': artifact['engine'],
        'target': artifact['target'],
        'training_scope': artifact['training_scope'],
        'training_rows': artifact['training_rows'],
        'unique_codes': artifact['unique_codes'],
        'allowed_times': artifact['allowed_times'],
        'global_default': artifact['global_default'],
        'evaluation': evaluation,
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
