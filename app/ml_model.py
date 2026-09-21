import re
import math
import operator
from datetime import date
from pathlib import Path
from typing import List, Tuple

import numpy as np

from . import ml_model_base as _base
from .ml_model_base import *  # noqa: F401,F403 - API interna compatible
from .schemas import PrediccionRequest
from .tariff_evidence import annotate_documentary_evidence

MAX_SUPPORT_LENGTH = 500


def fragmentos_clinicos(texto: str) -> List[str]:
    """Devuelve substrings literales del hallazgo; nunca genera evidencia."""
    fragmentos = []
    for match in re.finditer(r"[^.!?\r\n]+(?:[.!?]+|$)", texto or ""):
        fragmento = match.group(0).strip()
        if len(fragmento) >= 12:
            fragmentos.append(fragmento[:MAX_SUPPORT_LENGTH])
    return fragmentos


def _soportes_tfidf(artefacto: dict, fragmentos: List[str], codigos: List[str]) -> dict[str, str]:
    probabilidades = artefacto["classifier"].predict_proba(
        artefacto["vectorizer"].transform(fragmentos)
    )
    clases = [_base.normalizar_codigo(codigo) for codigo in artefacto["label_binarizer"].classes_]
    indices = {codigo: indice for indice, codigo in enumerate(clases)}
    umbral = float(artefacto.get("evidence_threshold", artefacto.get("threshold", 0.5)))
    soportes = {}
    for codigo in codigos:
        indice = indices.get(codigo)
        if indice is None:
            continue
        indice_fragmento = int(np.argmax(probabilidades[:, indice]))
        if float(probabilidades[indice_fragmento, indice]) >= umbral:
            soportes[codigo] = fragmentos[indice_fragmento]
    return soportes


def _soportes_transformer(artefacto: dict, fragmentos: List[str], codigos: List[str]) -> dict[str, str]:
    model_dir = _base.resolver_model_dir(artefacto)
    tokenizer, model, device, torch = _base.cargar_transformer_multilabel(
        str(model_dir), _base.settings.embedding_device
    )
    encoded = tokenizer(
        fragmentos,
        truncation=True,
        max_length=int(artefacto.get("max_length", 512)),
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        probabilidades = torch.sigmoid(model(**encoded).logits).detach().cpu().numpy()
    clases = [_base.normalizar_codigo(codigo) for codigo in artefacto["classes"]]
    indices = {codigo: indice for indice, codigo in enumerate(clases)}
    umbral = float(artefacto.get("evidence_threshold", artefacto.get("threshold", 0.5)))
    soportes = {}
    for codigo in codigos:
        indice = indices.get(codigo)
        if indice is None:
            continue
        indice_fragmento = int(np.argmax(probabilidades[:, indice]))
        if float(probabilidades[indice_fragmento, indice]) >= umbral:
            soportes[codigo] = fragmentos[indice_fragmento]
    return soportes


def soportes_por_codigo(artefacto: dict, ranking: List[dict], hallazgo: str) -> dict[str, str]:
    fragmentos = fragmentos_clinicos(hallazgo)
    codigos = [str(item.get("codigo") or "") for item in ranking]
    if not fragmentos or not codigos:
        return {}
    try:
        if artefacto.get("model") == "tfidf_logistic_regression_multilabel":
            return _soportes_tfidf(artefacto, fragmentos, codigos)
        if artefacto.get("model") == "transformer_multilabel":
            return _soportes_transformer(artefacto, fragmentos, codigos)
    except (IndexError, KeyError, ValueError, RuntimeError, TypeError):
        return {}
    return {}


def evidencia_no_disponible(nombre: str, fuente: str) -> dict:
    return {
        "disponible": False,
        "texto_soporte": None,
        "justificacion": (
            f"No hay evidencia textual ni puntaje independiente disponible para {nombre}. "
            "Requiere validacion del auditor."
        ),
        "score_ranking": None,
        "porcentaje_ranking": None,
        "fuente": fuente,
    }


def anexar_justificaciones_codigo(
    codigo_ranking: List[dict],
    hallazgo: str,
    soportes: dict[str, str] | None = None,
) -> List[dict]:
    """Añade explicación por código manteniendo score como ranking no calibrado."""
    soportes = soportes or {}
    fragmentos_validos = fragmentos_clinicos(hallazgo)
    resultado = []
    for item_original in codigo_ranking:
        item = dict(item_original)
        codigo = str(item.get("codigo") or "")
        score = float(item.get("score") or 0.0)
        texto_soporte = soportes.get(codigo)
        if texto_soporte not in fragmentos_validos:
            texto_soporte = None
        item.setdefault("descripcion_codigo", "")
        item["texto_soporte"] = texto_soporte
        if texto_soporte:
            item["justificacion"] = (
                f"Se sugirio el codigo {codigo} porque el modelo lo ubico en el ranking "
                f"con un puntaje no calibrado de {score:.4f}. El fragmento del hallazgo "
                f"que supero el umbral de evidencia para este codigo fue: "
                f"\"{texto_soporte}\". Requiere validacion del auditor."
            )
        else:
            item["justificacion"] = (
                f"Se sugirio el codigo {codigo} por su posicion en el ranking del modelo "
                f"con un puntaje no calibrado de {score:.4f}, pero no se encontro evidencia "
                "textual especifica suficiente para justificarlo. Requiere validacion del auditor."
            )
        item["evidencias"] = {
            "codigo": {
                "disponible": bool(texto_soporte),
                "texto_soporte": texto_soporte,
                "justificacion": item["justificacion"],
                "score_ranking": round(score, SCORE_DECIMALS),
                "porcentaje_ranking": round(score * 100.0, 4),
                "semantica_score": (
                    "Puntaje del modelo de codigos usado para ordenar sugerencias; "
                    "no es F1 ni probabilidad calibrada."
                ),
                "fuente": "modelo_codigos",
                "valor_sugerido": codigo,
            },
            "honorario": evidencia_no_disponible("el honorario", "modelo_honorarios"),
            "tiempo_anestesia": evidencia_no_disponible(
                "el tiempo de anestesia", "modelo_tiempo_anestesia"
            ),
        }
        resultado.append(item)
    return resultado


def metricas_modelo_desde_artefacto(artefacto: dict) -> dict | None:
    """Lee únicamente métricas holdout explícitas ligadas al artefacto activo."""
    # INICIO CAMBIO AUDITORIA TARIFARIO: métricas sintéticas no representan calidad clínica.
    if (
        artefacto.get("dataset_kind") == "synthetic"
        or artefacto.get("clinical_performance_validated", False) is not True
        or artefacto.get("clinical_validation_gate") != "approved_independent_operator_review"
    ):
        return None
    # FIN CAMBIO AUDITORIA TARIFARIO: métricas sintéticas no representan calidad clínica.
    if artefacto.get("training_scope") != "train_split":
        return None
    evaluacion = artefacto.get("evaluation") or {}
    metricas = evaluacion.get("final_test_metrics") or {}
    if evaluacion.get("evaluated") is not True:
        return None
    if not all(campo in metricas for campo in ("f1_macro", "f1_weighted", "size", "dataset")):
        return None
    version = artefacto.get("model_version")
    fecha = artefacto.get("evaluated_at")
    if not version or not fecha:
        return None
    # INICIO CAMBIO AUDITORIA TARIFARIO: sólo holdout final, finito y con procedencia explícita.
    try:
        if metricas["dataset"] != "test_holdout" or isinstance(metricas["size"], bool):
            return None
        if any(isinstance(metricas[name], bool) for name in ("f1_macro", "f1_weighted")):
            return None
        size = operator.index(metricas["size"])
        macro = float(metricas["f1_macro"])
        weighted = float(metricas["f1_weighted"])
        if size < 1 or not all(
            math.isfinite(value) and 0 <= value <= 1 for value in (macro, weighted)
        ):
            return None
        date.fromisoformat(str(fecha)[:10])
        if not isinstance(version, str) or not version.strip():
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    # FIN CAMBIO AUDITORIA TARIFARIO: sólo holdout final, finito y con procedencia explícita.
    return {
        "f1_macro": float(metricas["f1_macro"]),
        "f1_weighted": float(metricas["f1_weighted"]),
        "version_modelo": str(version),
        "conjunto_evaluacion": str(metricas["dataset"]),
        "fecha_evaluacion": str(fecha)[:10],
        "cantidad_muestras": int(metricas["size"]),
    }


def predecir_con_modelo(
    req: PrediccionRequest,
    path: Path = _base.MODEL_PATH,
) -> Tuple[dict, float]:
    artefacto = _base.cargar_modelo(path)
    if not artefacto:
        raise ValueError("Modelo ML no entrenado.")
    # INICIO CAMBIO AUDITORIA TARIFARIO: constructor compartido sólo por contrato de artefacto.
    candidate = _base.es_artefacto_tarifario(artefacto)
    texto = _base.texto_request(req, artefacto)
    if not texto and candidate and artefacto.get("allow_abstention") is True:
        prediccion = annotate_documentary_evidence(
            req, _base.prediccion_candidata_vacia(), candidate=True
        )
        prediccion.update(
            {
                "feature_schema_version": artefacto["feature_schema_version"],
                "feature_mode": artefacto.get("feature_mode", "clinical_document"),
            }
        )
        return prediccion, 0.0
    # FIN CAMBIO AUDITORIA TARIFARIO: constructor compartido sólo por contrato de artefacto.
    if not texto:
        raise ValueError("No hay texto clinico suficiente para predecir.")

    model_type = artefacto.get("model")
    if model_type == "tfidf_logistic_regression_multilabel":
        prediccion, score = _base.predecir_multilabel(artefacto, texto)
    elif model_type == "transformer_multilabel":
        prediccion, score = _base.predecir_transformer_multilabel(artefacto, texto)
    else:
        prediccion, score = _base.predecir_vecino(artefacto, texto)

    ranking = prediccion.get("codigo_ranking") or []
    hallazgo = req.hallazgos_conclusion or ""
    prediccion["codigo_ranking"] = anexar_justificaciones_codigo(
        ranking, hallazgo, soportes_por_codigo(artefacto, ranking, hallazgo)
    )
    metricas = metricas_modelo_desde_artefacto(artefacto)
    if metricas is not None:
        prediccion["metricas_modelo"] = metricas
    # INICIO CAMBIO AUDITORIA TARIFARIO: PDF se compara después del ranking y nunca lo repondera.
    prediccion = annotate_documentary_evidence(req, prediccion, candidate=candidate)
    if candidate:
        prediccion.update(
            {
                "feature_schema_version": artefacto["feature_schema_version"],
                "feature_mode": artefacto.get("feature_mode", "clinical_document"),
            }
        )
    # FIN CAMBIO AUDITORIA TARIFARIO: PDF se compara después del ranking y nunca lo repondera.
    return prediccion, score
