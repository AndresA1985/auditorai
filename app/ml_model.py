import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib

from .schemas import PrediccionRequest

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "auditai_model.joblib"


def separar_codigos(valor: str) -> List[str]:
    if not valor:
        return []
    codigos = []
    for parte in re.split(r"[,+]", str(valor)):
        codigo = parte.strip()
        if codigo and codigo not in codigos:
            codigos.append(codigo)
    return codigos


def parse_honorarios(valor: str, codigos: Iterable[str]) -> Dict[str, str]:
    honorarios: Dict[str, str] = {}
    if valor:
        for codigo, porcentaje in re.findall(r"([^(),+]+)\(([^()]*)\)", str(valor)):
            codigo = codigo.strip()
            porcentaje = porcentaje.strip()
            if codigo and porcentaje:
                honorarios[codigo] = porcentaje
    for codigo in codigos:
        honorarios.setdefault(codigo, "100")
    return honorarios


def texto_request(req: PrediccionRequest) -> str:
    return " ".join([
        req.procedimiento_sistema or "",
        req.hallazgos_conclusion or "",
        req.descripcion_estudio_013 or "",
    ]).strip()


def cargar_modelo(path: Path = MODEL_PATH):
    if not path.exists():
        return None
    return joblib.load(path)


def predecir_con_modelo(req: PrediccionRequest, path: Path = MODEL_PATH) -> Tuple[dict, float]:
    artefacto = cargar_modelo(path)
    if not artefacto:
        raise ValueError("Modelo ML no entrenado.")

    texto = texto_request(req)
    if not texto:
        raise ValueError("No hay texto clinico suficiente para predecir.")

    vectorizer = artefacto["vectorizer"]
    neighbors = artefacto["neighbors"]
    labels = artefacto["labels"]
    min_similarity = artefacto.get("min_similarity", 0.20)

    distances, indices = neighbors.kneighbors(vectorizer.transform([texto]), n_neighbors=1)
    similarity = 1.0 - float(distances[0][0])
    if similarity < min_similarity:
        raise ValueError("No se encontro una referencia historica suficientemente parecida.")

    label = labels[int(indices[0][0])]
    codigos = separar_codigos(label["codigo_grupo_auditor"])
    honorarios_codigo = parse_honorarios(label["honorario_auditor"], codigos)
    honorario = ",".join("{0}({1})".format(codigo, honorarios_codigo[codigo]) for codigo in codigos)

    return {
        "codigos": codigos,
        "honorarios_codigo": honorarios_codigo,
        "honorario": honorario,
        "tiempo_anestesia": label.get("tiempo_anestesia") or "",
        "nombre_procedimiento": label.get("nombre_procedimiento") or "",
        "observacion_auditor": "Propuesta generada por IA; validar antes de guardar.",
    }, similarity
