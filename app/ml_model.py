import re
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Tuple

import joblib
from sklearn.neighbors import NearestNeighbors

from .config import settings
from .schemas import PrediccionRequest

MODEL_PATH = settings.model_path


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


def cargar_modelo(path: Path = MODEL_PATH):
    if not path.exists():
        return None
    return joblib.load(path)


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


def predecir_con_modelo(req: PrediccionRequest, path: Path = MODEL_PATH) -> Tuple[dict, float]:
    artefacto = cargar_modelo(path)
    if not artefacto:
        raise ValueError("Modelo ML no entrenado.")

    texto = texto_request(req)
    if not texto:
        raise ValueError("No hay texto clinico suficiente para predecir.")

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
        "honorarios_codigo": {},
        "honorario": "",
        "tiempo_anestesia": "",
        "nombre_procedimiento": "",
        "observacion_auditor": "Codigos propuestos por IA; validar antes de guardar.",
    }, similarity
