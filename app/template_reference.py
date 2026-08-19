import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .db import fetch_all

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE_PATH = ROOT_DIR / 'models' / 'auditai_template_reference.joblib'
DEFAULT_EXCLUDED_CODES = {'99204'}
DEFAULT_MIN_SIMILARITY = 0.18
DEFAULT_TEMPLATE_BONUS = 0.25
DEFAULT_TEMPLATE_LIMIT = 3
DEFAULT_RANKING_LIMIT = 15


def normalizar_codigo(valor: Any) -> str:
    if valor is None:
        return ''
    return re.sub(r'\D+', '', str(valor))


def normalizar_texto(valor: Any) -> str:
    if not valor:
        return ''
    return re.sub(r'\s+', ' ', str(valor)).strip()


def cargar_plantillas_desde_db(excluded_codes: set[str] | None = None) -> list[dict]:
    excluded_codes = excluded_codes or DEFAULT_EXCLUDED_CODES
    sql = '''
        SELECT
            pl.codigo AS cod_plantilla,
            MAX(pl.descripcion) AS plantilla_descripcion,
            MAX(pl.desc_comp) AS plantilla_desc_comp,
            proc.tipo AS tipo,
            proc.codigo AS procedimiento_codigo,
            MAX(proc.descripcion) AS procedimiento_descripcion,
            SUM(pi.cantidad) AS cantidad_total,
            MIN(pi.orden) AS orden
        FROM ap_plantilla pl
        JOIN ap_plantilla_items pi ON pi.cod_plantilla = pl.codigo
        JOIN ap_procedimiento proc ON proc.id = pi.procedimiento
        WHERE pl.estado = 1
          AND pi.estado = 1
          AND proc.tipo = 'P'
          AND proc.codigo IS NOT NULL
          AND proc.codigo <> ''
        GROUP BY
            pl.codigo,
            proc.tipo,
            proc.codigo
        ORDER BY pl.codigo ASC, orden ASC, proc.codigo ASC
    '''
    rows = fetch_all(sql)
    by_template: dict[str, dict] = {}
    for row in rows:
        template_id = str(row.get('cod_plantilla') or '').strip()
        code = normalizar_codigo(row.get('procedimiento_codigo'))
        if not template_id or not code or code in excluded_codes:
            continue
        template = by_template.setdefault(template_id, {
            'cod_plantilla': template_id,
            'descripcion': normalizar_texto(row.get('plantilla_descripcion')),
            'desc_comp': normalizar_texto(row.get('plantilla_desc_comp')),
            'codigos': [],
            'procedimientos': [],
        })
        if code not in template['codigos']:
            template['codigos'].append(code)
        template['procedimientos'].append({
            'codigo': code,
            'tipo': row.get('tipo') or '',
            'descripcion': normalizar_texto(row.get('procedimiento_descripcion')),
            'cantidad': str(row.get('cantidad_total') or ''),
            'orden': row.get('orden'),
        })
    return list(by_template.values())


def texto_plantilla(template: dict) -> str:
    parts = [
        template.get('descripcion') or '',
        template.get('desc_comp') or '',
    ]
    for proc in template.get('procedimientos') or []:
        parts.append(proc.get('codigo') or '')
        parts.append(proc.get('descripcion') or '')
    return ' '.join(parts).strip()


def construir_referencia_plantillas(templates: list[dict], max_features: int = 100000) -> dict:
    texts = [texto_plantilla(template) for template in templates]
    pairs = [(template, text) for template, text in zip(templates, texts) if text and template.get('codigos')]
    if not pairs:
        raise RuntimeError('No hay plantillas activas con procedimientos tipo P para indexar.')
    templates = [template for template, _ in pairs]
    texts = [text for _, text in pairs]
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        strip_accents='unicode',
        lowercase=True,
        min_df=1,
        max_features=max_features,
    )
    matrix = vectorizer.fit_transform(texts)
    return {
        'version': 1,
        'artifact': 'template_reference',
        'target': 'codigo_grupo_auditor',
        'templates': templates,
        'texts': texts,
        'vectorizer': vectorizer,
        'matrix': matrix,
        'excluded_codes': sorted(DEFAULT_EXCLUDED_CODES),
    }


@lru_cache(maxsize=2)
def cargar_referencia_plantillas(path: str = str(DEFAULT_TEMPLATE_PATH)):
    template_path = Path(path)
    if not template_path.exists():
        return None
    return joblib.load(template_path)


def buscar_plantillas(texto: str, path: Path = DEFAULT_TEMPLATE_PATH, limit: int = DEFAULT_TEMPLATE_LIMIT) -> list[dict]:
    artifact = cargar_referencia_plantillas(str(path))
    if not artifact or not texto.strip():
        return []
    query = artifact['vectorizer'].transform([texto])
    scores = cosine_similarity(query, artifact['matrix'])[0]
    ordered = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)[:limit]
    matches = []
    for idx, score in ordered:
        template = artifact['templates'][int(idx)]
        matches.append({
            'cod_plantilla': template.get('cod_plantilla'),
            'descripcion': template.get('descripcion'),
            'desc_comp': template.get('desc_comp'),
            'codigos': list(template.get('codigos') or []),
            'score': round(float(score), 4),
        })
    return matches


def puntajes_plantillas(texto: str, min_similarity: float = DEFAULT_MIN_SIMILARITY, limit: int = DEFAULT_TEMPLATE_LIMIT) -> tuple[dict[str, float], list[dict]]:
    matches = buscar_plantillas(texto, limit=limit)
    code_scores: dict[str, float] = {}
    accepted = []
    for match in matches:
        score = float(match.get('score') or 0.0)
        if score < min_similarity:
            continue
        accepted.append(match)
        for code in match.get('codigos') or []:
            code_scores[code] = max(code_scores.get(code, 0.0), score)
    return code_scores, accepted


def anexar_soporte_plantillas(texto: str, prediccion: dict, score: float) -> tuple[dict, float]:
    _, matches = puntajes_plantillas(texto)
    prediccion = dict(prediccion)
    prediccion['plantilla_ranking'] = matches
    if matches:
        prediccion['observacion_auditor'] = (
            prediccion.get('observacion_auditor') or ''
        ).rstrip('.') + '; plantillas historicas disponibles solo como soporte visual.'
    return prediccion, score


# Compatibilidad con llamadas anteriores: ahora solo adjunta soporte visual.
def enriquecer_prediccion_con_plantillas(texto: str, prediccion: dict, score: float) -> tuple[dict, float]:
    return anexar_soporte_plantillas(texto, prediccion, score)
