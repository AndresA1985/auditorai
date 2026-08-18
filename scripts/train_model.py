import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from app.config import settings
from app.db import fetch_all


VALID_ENGINES = {"tfidf", "embeddings"}


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
            MAX(honorario_auditor) AS honorario_auditor,
            MAX(nombre_procedimiento) AS nombre_procedimiento,
            MAX(tiempo_anestesia) AS tiempo_anestesia,
            COUNT(*) AS detalles
        FROM ap_auditoria_doctor_detalle
        WHERE estado = 1
          AND hallazgo IS NOT NULL
          AND hallazgo <> ''
          AND codigo_grupo_auditor IS NOT NULL
          AND codigo_grupo_auditor <> ''
        GROUP BY
            id_agenda,
            codigo_grupo_auditor
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


def separar_codigos(valor: str) -> tuple[str, ...]:
    if not valor:
        return ()
    codigos = []
    for parte in str(valor).replace("+", ",").split(","):
        codigo = parte.strip()
        if codigo and codigo not in codigos:
            codigos.append(codigo)
    return tuple(codigos)


def etiqueta(row: dict) -> dict:
    return {
        "id_agenda": row.get("id_agenda"),
        "fecha": str(row.get("fecha") or ""),
        "id_empresa": row.get("id_empresa"),
        "id_seguro": row.get("id_seguro"),
        "codigo_grupo_auditor": row.get("codigo_grupo_auditor") or "",
    }


def clave_codigo(label: dict) -> tuple[str, ...]:
    return separar_codigos(label["codigo_grupo_auditor"])


def validar_split(validation_size: float, test_size: float) -> None:
    if validation_size <= 0 or test_size <= 0:
        raise RuntimeError("validation_size y test_size deben ser mayores que 0.")
    if validation_size + test_size >= 1:
        raise RuntimeError("validation_size + test_size debe ser menor que 1.")


def dividir_indices(total: int, validation_size: float, test_size: float) -> dict[str, list[int]]:
    validar_split(validation_size, test_size)
    idx = list(range(total))
    holdout_size = validation_size + test_size
    train_idx, holdout_idx = train_test_split(idx, test_size=holdout_size, random_state=42)
    relative_test_size = test_size / holdout_size
    validation_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=relative_test_size,
        random_state=42,
    )
    return {"train": train_idx, "validation": validation_idx, "test": test_idx}


def seleccionar(items: list, indices: list[int]) -> list:
    return [items[i] for i in indices]


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


def generar_embeddings(model, texts: list[str], batch_size: int):
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def construir_tfidf(texts: list[str]):
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
    return {"vectorizer": vectorizer, "neighbors": neighbors}


def construir_embeddings(texts: list[str], model_name: str, device: str, batch_size: int):
    model = cargar_sentence_transformer(model_name, device)
    matrix = generar_embeddings(model, texts, batch_size)
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    neighbors.fit(matrix)
    return {"embedding_model_name": model_name, "neighbors": neighbors}


def construir_modelo(texts: list[str], engine: str, model_name: str, device: str, batch_size: int):
    if engine == "tfidf":
        return construir_tfidf(texts)
    if engine == "embeddings":
        return construir_embeddings(texts, model_name, device, batch_size)
    raise ValueError("Motor de modelo no soportado: {0}".format(engine))


def calcular_metricas(eval_labels: list[dict], train_labels: list[dict], indices, distances) -> dict:
    y_true = [clave_codigo(label) for label in eval_labels]
    y_pred = [clave_codigo(train_labels[int(i[0])]) for i in indices]
    similarities = [1.0 - float(d[0]) for d in distances]

    exact_code_matches = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    code_overlap_scores = []
    for true, pred in zip(y_true, y_pred):
        true_set = set(true)
        pred_set = set(pred)
        union = true_set | pred_set
        code_overlap_scores.append(len(true_set & pred_set) / len(union) if union else 0.0)

    return {
        "exact_code_group_accuracy": round(exact_code_matches / len(y_true), 4),
        "avg_code_overlap": round(sum(code_overlap_scores) / len(code_overlap_scores), 4),
        "avg_similarity": round(sum(similarities) / len(similarities), 4),
        "min_similarity": round(min(similarities), 4),
    }


def evaluar_particion(nombre: str, labels: list[dict], train_labels: list[dict], indices, distances) -> dict:
    return {
        "name": nombre,
        "size": len(labels),
        **calcular_metricas(labels, train_labels, indices, distances),
    }


def evaluar(
    texts: list[str],
    labels: list[dict],
    validation_size: float,
    test_size: float,
    engine: str,
    model_name: str,
    device: str,
    batch_size: int,
) -> dict:
    if len(texts) < 20:
        return {"evaluated": False, "reason": "dataset demasiado pequeno"}

    split_idx = dividir_indices(len(texts), validation_size, test_size)
    train_texts = seleccionar(texts, split_idx["train"])
    validation_texts = seleccionar(texts, split_idx["validation"])
    test_texts = seleccionar(texts, split_idx["test"])
    train_labels = seleccionar(labels, split_idx["train"])
    validation_labels = seleccionar(labels, split_idx["validation"])
    test_labels = seleccionar(labels, split_idx["test"])

    if engine == "tfidf":
        model_artifact = construir_tfidf(train_texts)
        x_validation = model_artifact["vectorizer"].transform(validation_texts)
        x_test = model_artifact["vectorizer"].transform(test_texts)
    else:
        encoder = cargar_sentence_transformer(model_name, device)
        x_train = generar_embeddings(encoder, train_texts, batch_size)
        model_artifact = {"neighbors": NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")}
        model_artifact["neighbors"].fit(x_train)
        x_validation = generar_embeddings(encoder, validation_texts, batch_size)
        x_test = generar_embeddings(encoder, test_texts, batch_size)

    validation_distances, validation_indices = model_artifact["neighbors"].kneighbors(x_validation, n_neighbors=1)
    test_distances, test_indices = model_artifact["neighbors"].kneighbors(x_test, n_neighbors=1)

    return {
        "evaluated": True,
        "target": "codigo_grupo_auditor",
        "engine": engine,
        "embedding_model": model_name if engine == "embeddings" else "",
        "split": {
            "train_size": len(train_texts),
            "validation_size": len(validation_texts),
            "test_size": len(test_texts),
            "train_fraction": round(len(train_texts) / len(texts), 4),
            "validation_fraction": round(len(validation_texts) / len(texts), 4),
            "test_fraction": round(len(test_texts) / len(texts), 4),
        },
        "validation_metrics": evaluar_particion(
            "validation",
            validation_labels,
            train_labels,
            validation_indices,
            validation_distances,
        ),
        "final_test_metrics": evaluar_particion(
            "test",
            test_labels,
            train_labels,
            test_indices,
            test_distances,
        ),
    }


def entrenar(
    output: Path,
    min_similarity: float,
    limit: int | None,
    validation_size: float,
    test_size: float,
    engine: str,
    embedding_model: str,
    embedding_device: str,
    embedding_batch_size: int,
) -> dict:
    engine = engine.lower().strip()
    if engine not in VALID_ENGINES:
        raise RuntimeError("AUDITORIA_MODEL_ENGINE debe ser uno de: {0}".format(", ".join(sorted(VALID_ENGINES))))

    rows = cargar_dataset(limit)
    texts = [texto_entrenamiento(row) for row in rows]
    labels = [etiqueta(row) for row in rows]

    if not texts:
        raise RuntimeError("No hay registros completos para entrenar.")

    evaluation = evaluar(
        texts,
        labels,
        validation_size,
        test_size,
        engine,
        embedding_model,
        embedding_device,
        embedding_batch_size,
    )
    model_artifact = construir_modelo(texts, engine, embedding_model, embedding_device, embedding_batch_size)

    artifact = {
        "version": 4,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "engine": engine,
        "target": "codigo_grupo_auditor",
        "min_similarity": min_similarity,
        "labels": labels,
        "evaluation": evaluation,
        "metrics": evaluation.get("final_test_metrics", evaluation),
        "training_rows": len(rows),
        "unique_code_groups": len({clave_codigo(label) for label in labels}),
        **model_artifact,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    return {
        "output": str(output),
        "engine": artifact["engine"],
        "embedding_model": artifact.get("embedding_model_name", ""),
        "training_rows": artifact["training_rows"],
        "target": artifact["target"],
        "unique_code_groups": artifact["unique_code_groups"],
        "evaluation": evaluation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el modelo local auditai desde MySQL.")
    parser.add_argument("--output", default=str(settings.model_path))
    parser.add_argument("--min-similarity", type=float, default=settings.model_min_similarity)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validation-size", type=float, default=settings.validation_size)
    parser.add_argument("--test-size", type=float, default=settings.test_size)
    parser.add_argument("--engine", choices=sorted(VALID_ENGINES), default=settings.model_engine)
    parser.add_argument("--embedding-model", default=settings.embedding_model_name)
    parser.add_argument("--embedding-device", default=settings.embedding_device)
    parser.add_argument("--embedding-batch-size", type=int, default=settings.embedding_batch_size)
    args = parser.parse_args()

    result = entrenar(
        Path(args.output),
        args.min_similarity,
        args.limit,
        args.validation_size,
        args.test_size,
        args.engine,
        args.embedding_model,
        args.embedding_device,
        args.embedding_batch_size,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
