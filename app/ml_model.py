import re
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Tuple

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .config import settings
from .schemas import PrediccionRequest

MODEL_PATH = settings.model_path
DEFAULT_RANKING_LIMIT = 15


def separar_codigos(valor: str) -> List[str]:
    if not valor:
        return []
    codigos = []
    for parte in re.split(r"[,+]", str(valor)):
        codigo = parte.strip()
        if codigo and codigo not in codigos:
            codigos.append(codigo)
    return codigos


def texto_request(req: PrediccionRequest) -> str:
    return " ".join([
        req.procedimiento_sistema or "",
        req.hallazgos_conclusion or "",
        req.descripcion_estudio_013 or "",
    ]).strip()


@lru_cache(maxsize=2)
def cargar_sentence_transformer(model_name: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Falta instalar sentence-transformers. Ejecuta: pip install -r requirements.txt"
        ) from exc

    kwargs: dict[str, Any] = {}
    if device and device.lower() != "auto":
        kwargs["device"] = device
    return SentenceTransformer(model_name, **kwargs)


@lru_cache(maxsize=2)
def cargar_transformer_multilabel(model_dir: str, device: str):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Falta instalar torch/transformers. Ejecuta: pip install -r requirements.txt"
        ) from exc

    resolved_device = device
    if not resolved_device or resolved_device.lower() == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(resolved_device)
    model.eval()
    return tokenizer, model, resolved_device, torch


def cargar_modelo(path: Path = MODEL_PATH):
    if not path.exists():
        return None
    artefacto = joblib.load(path)
    if isinstance(artefacto, dict):
        artefacto["_artifact_path"] = str(path)
    return artefacto


def vectorizar_texto(artefacto: dict, texto: str):
    engine = artefacto.get("engine", "tfidf")
    if engine == "embeddings":
        model_name = artefacto.get("embedding_model_name") or settings.embedding_model_name
        model = cargar_sentence_transformer(model_name, settings.embedding_device)
        return model.encode(
            [texto],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    return artefacto["vectorizer"].transform([texto])


def binarizar_con_threshold(probabilities: np.ndarray, threshold: float, min_labels: int) -> np.ndarray:
    pred = probabilities >= threshold
    if min_labels > 0 and pred.sum() < min_labels:
        top_n = min(min_labels, probabilities.shape[0])
        top_idx = np.argpartition(probabilities, -top_n)[-top_n:]
        pred[top_idx] = True
    return pred


def predecir_multilabel(artefacto: dict, texto: str) -> Tuple[dict, float]:
    vector = artefacto["vectorizer"].transform([texto])
    probabilities = artefacto["classifier"].predict_proba(vector)[0]
    threshold = float(artefacto.get("threshold", 0.5))
    min_labels = int(artefacto.get("min_labels", 1))
    label_binarizer = artefacto["label_binarizer"]

    selected = binarizar_con_threshold(probabilities, threshold, min_labels)
    codigos = [str(codigo) for codigo, keep in zip(label_binarizer.classes_, selected) if keep]
    codigos.sort(key=lambda codigo: float(probabilities[list(label_binarizer.classes_).index(codigo)]), reverse=True)

    codigo_scores = {
        str(codigo): round(float(score), 4)
        for codigo, score in sorted(
            zip(label_binarizer.classes_, probabilities),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        if str(codigo) in codigos
    }
    codigo_ranking = [
        {"codigo": str(codigo), "score": round(float(score_item), 4), "selected": str(codigo) in codigos}
        for codigo, score_item in sorted(
            zip(label_binarizer.classes_, probabilities),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:DEFAULT_RANKING_LIMIT]
    ]
    score = max(codigo_scores.values()) if codigo_scores else 0.0

    return {
        "codigos": codigos,
        "codigo_scores": codigo_scores,
        "codigo_ranking": codigo_ranking,
        "honorarios_codigo": {},
        "honorario": "",
        "tiempo_anestesia": "",
        "nombre_procedimiento": "",
        "observacion_auditor": "Codigos propuestos por clasificador multi-label; validar antes de guardar.",
    }, score

def resolver_model_dir(artefacto: dict) -> Path:
    model_dir = Path(str(artefacto["model_dir"]))
    if model_dir.is_absolute():
        return model_dir
    artifact_path = Path(str(artefacto.get("_artifact_path") or MODEL_PATH))
    return artifact_path.parent / model_dir


def predecir_transformer_multilabel(artefacto: dict, texto: str) -> Tuple[dict, float]:
    model_dir = resolver_model_dir(artefacto)
    tokenizer, model, device, torch = cargar_transformer_multilabel(str(model_dir), settings.embedding_device)
    max_length = int(artefacto.get("max_length", 512))
    threshold = float(artefacto.get("threshold", 0.5))
    min_labels = int(artefacto.get("min_labels", 1))
    classes = artefacto["classes"]
    encoded = tokenizer(texto, truncation=True, max_length=max_length, padding=False, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        probabilities = torch.sigmoid(model(**encoded).logits[0]).detach().cpu().numpy()
    selected = binarizar_con_threshold(probabilities, threshold, min_labels)
    codigos = [str(codigo) for codigo, keep in zip(classes, selected) if keep]
    score_by_code = {str(codigo): round(float(score), 4) for codigo, score in zip(classes, probabilities)}
    codigos.sort(key=lambda codigo: score_by_code[codigo], reverse=True)
    codigo_scores = {codigo: score_by_code[codigo] for codigo in codigos}
    codigo_ranking = [
        {"codigo": str(codigo), "score": round(float(score_item), 4), "selected": bool(keep)}
        for codigo, score_item, keep in sorted(zip(classes, probabilities, selected), key=lambda item: float(item[1]), reverse=True)[:DEFAULT_RANKING_LIMIT]
    ]
    score = max(codigo_scores.values()) if codigo_scores else 0.0
    return {
        "codigos": codigos,
        "codigo_scores": codigo_scores,
        "codigo_ranking": codigo_ranking,
        "honorarios_codigo": {},
        "honorario": "",
        "tiempo_anestesia": "",
        "nombre_procedimiento": "",
        "observacion_auditor": "Codigos propuestos por Transformer fine-tuneado multi-label; validar antes de guardar.",
    }, score


def predecir_vecino(artefacto: dict, texto: str) -> Tuple[dict, float]:
    neighbors: NearestNeighbors = artefacto["neighbors"]
    labels = artefacto["labels"]
    min_similarity = artefacto.get("min_similarity", settings.model_min_similarity)

    vector = vectorizar_texto(artefacto, texto)
    distances, indices = neighbors.kneighbors(vector, n_neighbors=1)
    similarity = 1.0 - float(distances[0][0])
    if similarity < min_similarity:
        raise ValueError("No se encontro una referencia historica suficientemente parecida.")

    label = labels[int(indices[0][0])]
    codigos = separar_codigos(label["codigo_grupo_auditor"])

    return {
        "codigos": codigos,
        "codigo_scores": {},
        "codigo_ranking": [],
        "honorarios_codigo": {},
        "honorario": "",
        "tiempo_anestesia": "",
        "nombre_procedimiento": "",
        "observacion_auditor": "Codigos propuestos por IA; validar antes de guardar.",
    }, similarity


def predecir_con_modelo(req: PrediccionRequest, path: Path = MODEL_PATH) -> Tuple[dict, float]:
    artefacto = cargar_modelo(path)
    if not artefacto:
        raise ValueError("Modelo ML no entrenado.")

    texto = texto_request(req)
    if not texto:
        raise ValueError("No hay texto clinico suficiente para predecir.")

    model_type = artefacto.get("model")
    if model_type == "tfidf_logistic_regression_multilabel":
        return predecir_multilabel(artefacto, texto)
    if model_type == "transformer_multilabel":
        return predecir_transformer_multilabel(artefacto, texto)

    return predecir_vecino(artefacto, texto)
