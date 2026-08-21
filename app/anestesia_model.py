import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .schemas import PrediccionRequest

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ANESTESIA_PATH = ROOT_DIR / 'models' / 'auditai_anestesia.joblib'
DEFAULT_TIME = '0'


def normalizar_codigo(valor: Any) -> str:
    if valor is None:
        return ''
    return re.sub(r'\D+', '', str(valor))


def normalizar_tiempo(valor: Any) -> str:
    if valor is None or str(valor).strip() == '':
        return DEFAULT_TIME
    raw = str(valor).strip().replace(',', '.')
    try:
        number = float(raw)
    except ValueError:
        digits = re.sub(r'\D+', '', raw)
        return digits or DEFAULT_TIME
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
def cargar_modelo_anestesia(path: str = str(DEFAULT_ANESTESIA_PATH)):
    model_path = Path(path)
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@lru_cache(maxsize=2)
def cargar_transformer_runtime(model_dir: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return tokenizer, model, device


def predecir_tiempo_transformer(artifact: dict, texto: str, codigo: str) -> str:
    model_dir = artifact.get('model_dir')
    classes = [str(value) for value in artifact.get('classes') or []]
    if not model_dir or not classes:
        return DEFAULT_TIME

    tokenizer, model, device = cargar_transformer_runtime(model_dir)
    max_length = int(artifact.get('max_length') or 384)
    encoded = tokenizer(
        texto_codigo(texto, codigo),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt',
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
    label_idx = int(torch.argmax(logits, dim=1).item())
    if label_idx < 0 or label_idx >= len(classes):
        return DEFAULT_TIME
    return normalizar_tiempo(classes[label_idx])


def predecir_tiempo(artifact: dict, texto: str, codigo: str) -> str:
    code = normalizar_codigo(codigo)
    engine = artifact.get('engine', 'code_majority')
    default = str(artifact.get('global_default') or DEFAULT_TIME)
    by_code = artifact.get('by_code') or {}

    if artifact.get('model') == 'transformer_anestesia':
        return predecir_tiempo_transformer(artifact, texto, code)

    if engine == 'tfidf_logreg' and artifact.get('vectorizer') is not None:
        vector = artifact['vectorizer'].transform([texto_codigo(texto, code)])
        return normalizar_tiempo(artifact['classifier'].predict(vector)[0])

    return normalizar_tiempo(by_code.get(code) or default)


def tiempo_resumen(tiempos: dict[str, str]) -> str:
    valores = []
    for value in tiempos.values():
        try:
            valores.append(float(value))
        except (TypeError, ValueError):
            continue
    if not valores:
        return ''
    max_value = max(valores)
    if abs(max_value - round(max_value)) < 0.001:
        return str(int(round(max_value)))
    return str(round(max_value, 2)).rstrip('0').rstrip('.')


def aplicar_tiempos_anestesia(req: PrediccionRequest, prediccion: dict) -> dict:
    artifact = cargar_modelo_anestesia()
    codigos = [normalizar_codigo(codigo) for codigo in prediccion.get('codigos') or []]
    codigos = [codigo for codigo in codigos if codigo]

    prediccion = dict(prediccion)
    if not codigos:
        prediccion['tiempos_anestesia_codigo'] = {}
        prediccion['tiempo_anestesia'] = ''
        return prediccion

    texto = texto_request(req)
    tiempos = {}
    for codigo in codigos:
        tiempos[codigo] = predecir_tiempo(artifact, texto, codigo) if artifact else DEFAULT_TIME

    prediccion['tiempos_anestesia_codigo'] = tiempos
    prediccion['tiempo_anestesia'] = tiempo_resumen(tiempos)
    if artifact:
        prediccion['observacion_auditor'] = (
            prediccion.get('observacion_auditor') or ''
        ).rstrip('.') + '; tiempos de anestesia sugeridos por modelo historico.'
    return prediccion
