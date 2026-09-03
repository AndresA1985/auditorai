import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from .schemas import PrediccionRequest

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_HONORARIOS_PATH = ROOT_DIR / 'models' / 'auditai_honorarios.joblib'
DEFAULT_PERCENTAGE = '100'


def normalizar_codigo(valor: Any) -> str:
    if valor is None:
        return ''
    return re.sub(r'\D+', '', str(valor))


def normalizar_porcentaje(valor: Any) -> str:
    if valor is None or str(valor).strip() == '':
        return DEFAULT_PERCENTAGE
    raw = str(valor).strip().replace('%', '').replace(',', '.')
    try:
        number = float(raw)
    except ValueError:
        digits = re.sub(r'\D+', '', raw)
        return digits or DEFAULT_PERCENTAGE
    if 0 < number <= 1:
        number *= 100
    if abs(number - round(number)) < 0.001:
        return str(int(round(number)))
    return str(round(number, 2)).rstrip('0').rstrip('.')


def texto_request(req: PrediccionRequest) -> str:
    return ' '.join([
        req.procedimiento_sistema or '',
        req.hallazgos_conclusion or '',
        req.descripcion_estudio_013 or '',
    ]).strip()


def texto_codigo(texto: str, codigo: str) -> str:
    return 'CODIGO_{0} {1}'.format(normalizar_codigo(codigo), texto or '').strip()


@lru_cache(maxsize=2)
def cargar_modelo_honorarios(path: str = str(DEFAULT_HONORARIOS_PATH)):
    model_path = Path(path)
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def predecir_porcentaje(artifact: dict, texto: str, codigo: str) -> str:
    code = normalizar_codigo(codigo)
    engine = artifact.get('engine', 'code_majority')
    default = str(artifact.get('global_default') or DEFAULT_PERCENTAGE)
    by_code = artifact.get('by_code') or {}

    if engine == 'tfidf_logreg' and artifact.get('vectorizer') is not None:
        vector = artifact['vectorizer'].transform([texto_codigo(texto, code)])
        return str(artifact['classifier'].predict(vector)[0])

    return str(by_code.get(code) or default)


def evidencia_honorario(artifact: dict | None, texto: str, hallazgo: str, codigo: str) -> dict:
    code = normalizar_codigo(codigo)
    if not artifact or artifact.get('engine') != 'tfidf_logreg':
        valor = predecir_porcentaje(artifact, texto, code) if artifact else DEFAULT_PERCENTAGE
        return {
            'disponible': False,
            'texto_soporte': None,
            'justificacion': (
                f'El honorario sugerido para el codigo {code} es {valor}%, pero el modelo '
                'activo no genera un puntaje textual independiente. Requiere validacion del auditor.'
            ),
            'score_ranking': None,
            'porcentaje_ranking': None,
            'fuente': 'modelo_honorarios',
            'valor_sugerido': valor,
            'unidad': 'porcentaje_honorario',
        }

    classifier = artifact.get('classifier')
    vectorizer = artifact.get('vectorizer')
    if classifier is None or vectorizer is None or not hasattr(classifier, 'predict_proba'):
        return evidencia_honorario(None, texto, hallazgo, code)

    full_vector = vectorizer.transform([texto_codigo(texto, code)])
    probabilities = classifier.predict_proba(full_vector)[0]
    classes = [str(value) for value in classifier.classes_]
    selected_idx = int(probabilities.argmax())
    valor = normalizar_porcentaje(classes[selected_idx])
    score = float(probabilities[selected_idx])

    from .ml_model import fragmentos_clinicos
    fragmentos = fragmentos_clinicos(hallazgo)
    soporte = None
    if fragmentos:
        fragment_vectors = vectorizer.transform([texto_codigo(item, code) for item in fragmentos])
        fragment_probabilities = classifier.predict_proba(fragment_vectors)[:, selected_idx]
        best_fragment_idx = int(fragment_probabilities.argmax())
        evidence_threshold = float(artifact.get('evidence_threshold', 0.5))
        if float(fragment_probabilities[best_fragment_idx]) >= evidence_threshold:
            soporte = fragmentos[best_fragment_idx]

    porcentaje = round(score * 100.0, 4)
    if soporte:
        justificacion = (
            f'Para el codigo {code}, el modelo de honorarios sugirio {valor}% con un '
            f'puntaje de ranking no calibrado de {porcentaje:.2f}%. La evidencia literal '
            f'mas asociada fue: "{soporte}". Requiere validacion del auditor.'
        )
    else:
        justificacion = (
            f'Para el codigo {code}, el modelo de honorarios sugirio {valor}% con un '
            f'puntaje de ranking no calibrado de {porcentaje:.2f}%, calculado sobre el '
            'texto completo, pero no se encontro evidencia textual especifica suficiente '
            'en ningun fragmento. Requiere validacion del auditor.'
        )
    return {
        'disponible': bool(soporte),
        'texto_soporte': soporte,
        'justificacion': justificacion,
        'score_ranking': round(score, 6),
        'porcentaje_ranking': porcentaje,
        'semantica_score': (
            'Puntaje de ranking de la clase de honorario elegida; '
            'no es F1 ni probabilidad calibrada.'
        ),
        'fuente': 'modelo_honorarios',
        'valor_sugerido': valor,
        'unidad': 'porcentaje_honorario',
    }


def evidencia_honorario_segura(
    artifact: dict | None, texto: str, hallazgo: str, codigo: str
) -> dict:
    try:
        return evidencia_honorario(artifact, texto, hallazgo, codigo)
    except Exception:
        evidencia = evidencia_honorario(None, texto, hallazgo, codigo)
        evidencia['justificacion'] = (
            f'No fue posible calcular evidencia independiente de honorario para el codigo '
            f'{normalizar_codigo(codigo)}. Se aplico el valor fallback y requiere validacion '
            'del auditor.'
        )
        return evidencia


def aplicar_honorarios(req: PrediccionRequest, prediccion: dict) -> dict:
    artifact = cargar_modelo_honorarios()
    codigos = [normalizar_codigo(codigo) for codigo in prediccion.get('codigos') or []]
    codigos = [codigo for codigo in codigos if codigo]

    if not codigos:
        prediccion['honorarios_codigo'] = {}
        prediccion['honorario'] = ''
        return prediccion

    texto = texto_request(req)
    ranking = [dict(item) for item in prediccion.get('codigo_ranking') or []]
    ranking_codes = [normalizar_codigo(item.get('codigo')) for item in ranking]
    evidencias = {
        code: evidencia_honorario_segura(artifact, texto, req.hallazgos_conclusion or '', code)
        for code in ranking_codes if code
    }
    honorarios = {}
    for codigo in codigos:
        evidencia = evidencias.get(codigo)
        honorarios[codigo] = (
            str(evidencia.get('valor_sugerido')) if evidencia else
            (predecir_porcentaje(artifact, texto, codigo) if artifact else DEFAULT_PERCENTAGE)
        )

    prediccion = dict(prediccion)
    prediccion['codigos'] = codigos
    for item, code in zip(ranking, ranking_codes):
        bloques = dict(item.get('evidencias') or {})
        if code in evidencias:
            bloques['honorario'] = evidencias[code]
        item['evidencias'] = bloques
    prediccion['codigo_ranking'] = ranking
    prediccion['honorarios_codigo'] = honorarios
    prediccion['honorario'] = ','.join('{0}({1})'.format(codigo, honorarios[codigo]) for codigo in codigos)
    if artifact:
        prediccion['observacion_auditor'] = (
            prediccion.get('observacion_auditor') or ''
        ).rstrip('.') + '; honorarios sugeridos por modelo historico.'
    return prediccion
