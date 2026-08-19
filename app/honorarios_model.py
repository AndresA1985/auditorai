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


def aplicar_honorarios(req: PrediccionRequest, prediccion: dict) -> dict:
    artifact = cargar_modelo_honorarios()
    codigos = [normalizar_codigo(codigo) for codigo in prediccion.get('codigos') or []]
    codigos = [codigo for codigo in codigos if codigo]

    if not codigos:
        prediccion['honorarios_codigo'] = {}
        prediccion['honorario'] = ''
        return prediccion

    texto = texto_request(req)
    honorarios = {}
    for codigo in codigos:
        if artifact:
            honorarios[codigo] = predecir_porcentaje(artifact, texto, codigo)
        else:
            honorarios[codigo] = DEFAULT_PERCENTAGE

    prediccion = dict(prediccion)
    prediccion['codigos'] = codigos
    prediccion['honorarios_codigo'] = honorarios
    prediccion['honorario'] = ','.join('{0}({1})'.format(codigo, honorarios[codigo]) for codigo in codigos)
    if artifact:
        prediccion['observacion_auditor'] = (
            prediccion.get('observacion_auditor') or ''
        ).rstrip('.') + '; honorarios sugeridos por modelo historico.'
    return prediccion
