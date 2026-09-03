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


def evidencia_tiempo_anestesia(
    artifact: dict | None, texto: str, hallazgo: str, codigo: str
) -> dict:
    code = normalizar_codigo(codigo)
    if not artifact or (
        artifact.get('model') != 'transformer_anestesia'
        and artifact.get('engine') != 'tfidf_logreg'
    ):
        valor = predecir_tiempo(artifact, texto, code) if artifact else DEFAULT_TIME
        return {
            'disponible': False,
            'texto_soporte': None,
            'justificacion': (
                f'El tiempo de anestesia sugerido para el codigo {code} es {valor}, pero '
                'el modelo activo no genera un puntaje textual independiente. '
                'Requiere validacion del auditor.'
            ),
            'score_ranking': None,
            'porcentaje_ranking': None,
            'fuente': 'modelo_tiempo_anestesia',
            'valor_sugerido': valor,
            'unidad': 'horas',
        }

    from .ml_model import fragmentos_clinicos
    fragmentos = fragmentos_clinicos(hallazgo)
    soporte = None

    if artifact.get('model') == 'transformer_anestesia':
        model_dir = artifact.get('model_dir')
        classes = [str(value) for value in artifact.get('classes') or []]
        if not model_dir or not classes:
            return evidencia_tiempo_anestesia(None, texto, hallazgo, code)
        tokenizer, model, device = cargar_transformer_runtime(model_dir)
        max_length = int(artifact.get('max_length') or 384)

        def probabilities_for(items):
            encoded = tokenizer(
                [texto_codigo(item, code) for item in items],
                truncation=True, padding=True, max_length=max_length, return_tensors='pt',
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                return torch.softmax(model(**encoded).logits, dim=1).detach().cpu().numpy()

        full_probabilities = probabilities_for([texto])[0]
        selected_idx = int(full_probabilities.argmax())
        score = float(full_probabilities[selected_idx])
        valor = normalizar_tiempo(classes[selected_idx])
        if fragmentos:
            fragment_probabilities = probabilities_for(fragmentos)[:, selected_idx]
            best_fragment_idx = int(fragment_probabilities.argmax())
            evidence_threshold = float(artifact.get('evidence_threshold', 0.5))
            if float(fragment_probabilities[best_fragment_idx]) >= evidence_threshold:
                soporte = fragmentos[best_fragment_idx]
        fuente = 'modelo_tiempo_anestesia_transformer'
    else:
        classifier = artifact.get('classifier')
        vectorizer = artifact.get('vectorizer')
        if classifier is None or vectorizer is None or not hasattr(classifier, 'predict_proba'):
            return evidencia_tiempo_anestesia(None, texto, hallazgo, code)
        full_vector = vectorizer.transform([texto_codigo(texto, code)])
        full_probabilities = classifier.predict_proba(full_vector)[0]
        classes = [str(value) for value in classifier.classes_]
        selected_idx = int(full_probabilities.argmax())
        score = float(full_probabilities[selected_idx])
        valor = normalizar_tiempo(classes[selected_idx])
        if fragmentos:
            fragment_vectors = vectorizer.transform(
                [texto_codigo(item, code) for item in fragmentos]
            )
            fragment_probabilities = classifier.predict_proba(fragment_vectors)[:, selected_idx]
            best_fragment_idx = int(fragment_probabilities.argmax())
            evidence_threshold = float(artifact.get('evidence_threshold', 0.5))
            if float(fragment_probabilities[best_fragment_idx]) >= evidence_threshold:
                soporte = fragmentos[best_fragment_idx]
        fuente = 'modelo_tiempo_anestesia_tfidf'

    porcentaje = round(score * 100.0, 4)
    if soporte:
        justificacion = (
            f'Para el codigo {code}, el modelo de anestesia sugirio {valor} con un '
            f'puntaje de ranking no calibrado de {porcentaje:.2f}%. La evidencia literal '
            f'mas asociada fue: "{soporte}". Requiere validacion del auditor.'
        )
    else:
        justificacion = (
            f'Para el codigo {code}, el modelo de anestesia sugirio {valor} con un '
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
            'Puntaje de ranking de la clase de tiempo elegida; '
            'no es F1 ni probabilidad calibrada.'
        ),
        'fuente': fuente,
        'valor_sugerido': valor,
        'unidad': 'horas',
    }


def evidencia_tiempo_anestesia_segura(
    artifact: dict | None, texto: str, hallazgo: str, codigo: str
) -> dict:
    try:
        return evidencia_tiempo_anestesia(artifact, texto, hallazgo, codigo)
    except Exception:
        evidencia = evidencia_tiempo_anestesia(None, texto, hallazgo, codigo)
        evidencia['justificacion'] = (
            f'No fue posible calcular evidencia independiente de tiempo de anestesia para '
            f'el codigo {normalizar_codigo(codigo)}. Se aplico el valor fallback y requiere '
            'validacion del auditor.'
        )
        return evidencia


def observacion_por_codigos(ranking: list[dict]) -> str:
    def detalle(etiqueta: str, evidencia: dict, sufijo_valor: str) -> str:
        valor = evidencia.get('valor_sugerido') or 'no disponible'
        porcentaje = evidencia.get('porcentaje_ranking')
        ranking_text = (
            f'{float(porcentaje):.2f}%' if porcentaje is not None else 'no disponible'
        )
        soporte = evidencia.get('texto_soporte')
        soporte_text = (
            f'El fragmento literal mas relacionado fue: "{soporte}".'
            if soporte else
            'No se encontro evidencia textual especifica suficiente.'
        )
        return (
            f'{etiqueta}: {valor}{sufijo_valor}; puntaje de ranking sobre el texto '
            f'completo: {ranking_text}. {soporte_text}'
        )

    parrafos = []
    for item in ranking:
        if not item.get('selected'):
            continue
        codigo = str(item.get('codigo') or '')
        evidencias = item.get('evidencias') or {}
        codigo_justificacion = str(
            (evidencias.get('codigo') or {}).get('justificacion') or
            'No hay justificacion de codigo disponible.'
        )
        honorario = detalle(
            'Honorario sugerido', evidencias.get('honorario') or {}, ' %'
        )
        anestesia = detalle(
            'Tiempo sugerido', evidencias.get('tiempo_anestesia') or {}, ' horas'
        )
        parrafos.append(
            f'Explicacion integral para el codigo {codigo}. {codigo_justificacion} '
            f'{honorario} {anestesia} Los puntajes son relativos y no calibrados; '
            'toda la propuesta requiere validacion del auditor.'
        )
    return '\n\n'.join(parrafos)


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
    ranking = [dict(item) for item in prediccion.get('codigo_ranking') or []]
    ranking_codes = [normalizar_codigo(item.get('codigo')) for item in ranking]
    evidencias = {
        code: evidencia_tiempo_anestesia_segura(
            artifact, texto, req.hallazgos_conclusion or '', code
        )
        for code in ranking_codes if code
    }
    tiempos = {}
    for codigo in codigos:
        evidencia = evidencias.get(codigo)
        tiempos[codigo] = (
            str(evidencia.get('valor_sugerido')) if evidencia else
            (predecir_tiempo(artifact, texto, codigo) if artifact else DEFAULT_TIME)
        )

    for item, code in zip(ranking, ranking_codes):
        bloques = dict(item.get('evidencias') or {})
        if code in evidencias:
            bloques['tiempo_anestesia'] = evidencias[code]
        item['evidencias'] = bloques
    prediccion['codigo_ranking'] = ranking
    prediccion['tiempos_anestesia_codigo'] = tiempos
    prediccion['tiempo_anestesia'] = tiempo_resumen(tiempos)
    observacion = observacion_por_codigos(ranking)
    if observacion:
        prediccion['observacion_auditor'] = observacion
    return prediccion
