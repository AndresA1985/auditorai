import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from app.config import settings
from app.db import fetch_all


def cargar_dataset(limit: int | None = None) -> list[dict]:
    sql = """
        SELECT
            id_agenda,
            MAX(fecha) AS fecha,
            MAX(id_empresa) AS id_empresa,
            MAX(id_seguro) AS id_seguro,
            MAX(procedimiento_auditor) AS procedimiento_auditor,
            MAX(hallazgo) AS hallazgo,
            MAX(conclusion) AS conclusion,
            codigo_grupo_auditor,
            honorario_auditor,
            nombre_procedimiento,
            tiempo_anestesia,
            COUNT(*) AS detalles
        FROM ap_auditoria_doctor_detalle
        WHERE estado = 1
          AND hallazgo IS NOT NULL
          AND hallazgo <> ''
          AND codigo_grupo_auditor IS NOT NULL
          AND codigo_grupo_auditor <> ''
          AND honorario_auditor IS NOT NULL
          AND honorario_auditor <> ''
          AND nombre_procedimiento IS NOT NULL
          AND nombre_procedimiento <> ''
        GROUP BY
            id_agenda,
            codigo_grupo_auditor,
            honorario_auditor,
            nombre_procedimiento,
            tiempo_anestesia
        ORDER BY MAX(fecha) ASC, id_agenda ASC
    """
    if limit:
        sql += "\nLIMIT {0}".format(int(limit))
    return fetch_all(sql)


def texto_entrenamiento(row: dict) -> str:
    return " ".join([
        str(row.get("procedimiento_auditor") or ""),
        str(row.get("nombre_procedimiento") or ""),
        str(row.get("hallazgo") or ""),
        str(row.get("conclusion") or ""),
    ]).strip()


def etiqueta(row: dict) -> dict:
    return {
        "id_agenda": row.get("id_agenda"),
        "fecha": str(row.get("fecha") or ""),
        "id_empresa": row.get("id_empresa"),
        "id_seguro": row.get("id_seguro"),
        "codigo_grupo_auditor": row.get("codigo_grupo_auditor") or "",
        "honorario_auditor": row.get("honorario_auditor") or "",
        "nombre_procedimiento": row.get("nombre_procedimiento") or "",
        "tiempo_anestesia": row.get("tiempo_anestesia"),
    }


def clave_etiqueta(label: dict) -> tuple:
    return (
        label["codigo_grupo_auditor"],
        label["honorario_auditor"],
        label["nombre_procedimiento"],
        label["tiempo_anestesia"],
    )


def evaluar(texts: list[str], labels: list[dict], test_size: float) -> dict:
    if len(texts) < 20:
        return {"evaluated": False, "reason": "dataset demasiado pequeno"}

    idx = list(range(len(texts)))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=42)
    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        strip_accents="unicode",
        lowercase=True,
        min_df=2,
        max_features=250000,
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    neighbors.fit(x_train)
    distances, indices = neighbors.kneighbors(x_test, n_neighbors=1)

    y_true = [clave_etiqueta(label) for label in test_labels]
    y_pred = [clave_etiqueta(train_labels[int(i[0])]) for i in indices]
    similarities = [1.0 - float(d[0]) for d in distances]

    exact_matches = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)

    return {
        "evaluated": True,
        "test_size": len(test_idx),
        "exact_label_accuracy": round(exact_matches / len(y_true), 4),
        "avg_similarity": round(sum(similarities) / len(similarities), 4),
        "min_similarity": round(min(similarities), 4),
    }


def entrenar(output: Path, min_similarity: float, limit: int | None, test_size: float) -> dict:
    rows = cargar_dataset(limit)
    texts = [texto_entrenamiento(row) for row in rows]
    labels = [etiqueta(row) for row in rows]

    if not texts:
        raise RuntimeError("No hay registros completos para entrenar.")

    metrics = evaluar(texts, labels, test_size)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        strip_accents="unicode",
        lowercase=True,
        min_df=2,
        max_features=250000,
    )
    matrix = vectorizer.fit_transform(texts)
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    neighbors.fit(matrix)

    artifact = {
        "version": 1,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "min_similarity": min_similarity,
        "vectorizer": vectorizer,
        "neighbors": neighbors,
        "labels": labels,
        "metrics": metrics,
        "training_rows": len(rows),
        "unique_labels": len({clave_etiqueta(label) for label in labels}),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    return {
        "output": str(output),
        "training_rows": artifact["training_rows"],
        "unique_labels": artifact["unique_labels"],
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el modelo local auditai desde MySQL.")
    parser.add_argument("--output", default=str(settings.model_path))
    parser.add_argument("--min-similarity", type=float, default=settings.model_min_similarity)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=settings.train_test_size)
    args = parser.parse_args()

    result = entrenar(Path(args.output), args.min_similarity, args.limit, args.test_size)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
