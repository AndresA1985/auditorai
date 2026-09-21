# INICIO CAMBIO AUDITORAI TARIFARIO: snapshots previos y particiones sin fuga.
"""Pure snapshot validation. No database, configuration or model imports."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys

SCHEMA_VERSION = "tariff_snapshot_v1"
MAX_DATASET_BYTES = 128 * 1024 * 1024
FORBIDDEN_FIELDS = {
    "cedula", "paciente", "nombre", "filename", "archivo", "codigo_validacion",
    "id_agenda", "id_hc", "id_empresa", "id_seguro", "fecha_nacimiento",
    "procedimiento_auditor", "nombre_procedimiento", "observacion_auditor",
    "codigo_grupo_auditor", "honorario_auditor", "tiempo_anestesia",
}
TEXT_FIELDS = ("procedimiento_sistema", "hallazgos_conclusion", "descripcion_estudio_013", "informe_tecnico_justificacion")
RECORD_FIELDS = set(TEXT_FIELDS) | {
    "sample_id", "patient_group", "admission_group", "snapshot_immutable", "source_revision", "provenance_review_id", "label_origin",
    "prediction_at", "clinical_snapshot_captured_at", "clinical_available_at", "label_reviewed_at", "technical_available_at",
    "evidencia_tarifario", "labels", "document_available_at", "document_snapshot_captured_at", "document_hash_verified_at", "document_same_admission_verified",
}
EVIDENCE_FIELDS = {"estado", "fuente", "codigos", "advertencias", "documento_sha256"}
CODE_FIELDS = {"codigo", "descripcion", "origen", "pagina", "confianza_ocr"}
GROUP = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,99}\Z")
HASH = re.compile(r"[a-fA-F0-9]{64}\Z")
CODE = re.compile(r"[0-9]{5,6}\Z")
PERSONAL_TOKEN = re.compile(r"(?<!\d)\d{10}(?!\d)|\bCV[A-Za-z0-9]{5,}\b|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", re.I)
ALLOWED_WARNINGS = {
    "sin_documento", "sin_archivo", "pdf_no_adjuntado", "pdf_encriptado",
    "pdf_sin_paginas", "tiempo_extraccion_agotado", "ocr_no_disponible",
    "ocr_sin_codigos_verificables", "requiere_verificacion_ocr",
    "tiempo_ocr_agotado", "ocr_fallido", "limite_codigos",
    "sin_codigos_tarifarios", "pdf_invalido", "extraccion_fallida",
}


class SnapshotError(ValueError):
    """Messages deliberately contain no row contents or identifiers."""


def timestamp(value) -> datetime:
    if not isinstance(value, str):
        raise SnapshotError("timestamp_invalido")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SnapshotError("timestamp_invalido") from error
    if result.tzinfo is None:
        raise SnapshotError("timestamp_sin_zona")
    return result.astimezone(timezone.utc)


def validate_record(row: dict, *, data_kind: str) -> str | None:
    if not isinstance(row, dict):
        return "fila_invalida"
    if FORBIDDEN_FIELDS.intersection(row):
        return "identidad_o_decision_en_features"
    if set(row) - RECORD_FIELDS:
        return "campos_no_permitidos"
    if not all(isinstance(row.get(key), str) and GROUP.fullmatch(row[key]) for key in ("sample_id", "patient_group", "admission_group")):
        return "seudonimos_invalidos"
    if row.get("snapshot_immutable") is not True:
        return "snapshot_no_inmutable"
    if data_kind == "clinical" and (row.get("source_revision") not in {"verified_archive", "prospective_capture"} or not row.get("provenance_review_id")):
        return "procedencia_no_revisada"
    if data_kind == "clinical" and row.get("label_origin") != "auditor_reviewed":
        return "etiqueta_no_revisada"
    try:
        predicted = timestamp(row.get("prediction_at"))
        decided = timestamp(row.get("label_reviewed_at"))
        captured = timestamp(row.get("clinical_snapshot_captured_at"))
        available = timestamp(row.get("clinical_available_at"))
        if not available <= captured <= predicted <= decided:
            return "evidencia_clinica_posterior"
        if row.get("informe_tecnico_justificacion") and timestamp(row.get("technical_available_at")) > predicted:
            return "informe_posterior"
    except SnapshotError as error:
        return str(error)
    if not any(str(row.get(field) or "").strip() for field in TEXT_FIELDS[:3]):
        return "clinica_ausente"
    for field in TEXT_FIELDS:
        text = row.get(field) or ""
        if not isinstance(text, str) or len(text) > 20000:
            return "texto_invalido"
        if PERSONAL_TOKEN.search(text):
            return "texto_no_desidentificado"
    labels = row.get("labels")
    if not isinstance(labels, list) or not labels or any(not isinstance(code, str) or not CODE.fullmatch(code) for code in labels):
        return "etiquetas_invalidas"
    evidence = row.get("evidencia_tarifario")
    if evidence is not None:
        if not isinstance(evidence, dict) or set(evidence) - EVIDENCE_FIELDS or not isinstance(evidence.get("codigos", []), list):
            return "evidencia_invalida"
        if evidence.get("estado") not in {"extraido", "sin_codigos", "ilegible", "sin_documento", "sin_archivo", "sin_pdf"} or evidence.get("fuente") not in {"texto", "ocr", "mixta", "ninguna"}:
            return "estado_documental_invalido"
        warnings = evidence.get("advertencias", [])
        if not isinstance(warnings, list) or any(warning not in ALLOWED_WARNINGS for warning in warnings):
            return "advertencias_invalidas"
        codes = evidence.get("codigos", [])
        if (evidence.get("estado") == "extraido") != bool(codes):
            return "estado_documental_invalido"
        if len(codes) > 25:
            return "demasiados_tarifarios"
        origins = set()
        for item in codes:
            if not isinstance(item, dict) or set(item) - CODE_FIELDS or not CODE.fullmatch(str(item.get("codigo") or "")):
                return "tarifario_invalido"
            if item.get("origen") not in {"texto", "ocr"} or isinstance(item.get("pagina"), bool) or not isinstance(item.get("pagina"), int) or not 1 <= item["pagina"] <= 10:
                return "origen_tarifario_invalido"
            origins.add(item["origen"])
            confidence = item.get("confianza_ocr")
            if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 100):
                return "confianza_ocr_invalida"
            description = item.get("descripcion") or ""
            if not isinstance(description, str) or len(description) > 180 or PERSONAL_TOKEN.search(description):
                return "descripcion_no_desidentificada"
        expected_source = "mixta" if len(origins) > 1 else next(iter(origins), "ninguna")
        if evidence.get("fuente") != expected_source:
            return "fuente_documental_invalida"
        digest = evidence.get("documento_sha256")
        document_present = evidence.get("estado") in {"extraido", "sin_codigos", "ilegible"}
        if document_present and not digest:
            return "hash_documental_ausente"
        if not document_present and digest:
            return "hash_documental_sin_pdf"
        if document_present:
            if not isinstance(digest, str) or not HASH.fullmatch(digest):
                return "hash_documental_invalido"
            if row.get("document_same_admission_verified") is not True:
                return "documento_no_corresponde"
            try:
                available_doc = timestamp(row.get("document_available_at"))
                captured_doc = timestamp(row.get("document_snapshot_captured_at"))
                verified_doc = timestamp(row.get("document_hash_verified_at"))
                if not available_doc <= captured_doc <= verified_doc <= predicted:
                    return "evidencia_documental_posterior"
            except SnapshotError as error:
                return str(error)
    return None


def load_snapshot_dataset(path: str | Path) -> dict:
    source = Path(path)
    if source.stat().st_size > MAX_DATASET_BYTES:
        raise SnapshotError("dataset_demasiado_grande")
    raw = source.read_bytes()
    try:
        document = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise SnapshotError("dataset_json_invalido") from error
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION or document.get("deidentified") is not True:
        raise SnapshotError("dataset_sin_contrato_desidentificado")
    kind = document.get("data_kind")
    if kind not in {"clinical", "synthetic"} or not isinstance(document.get("records"), list):
        raise SnapshotError("dataset_invalido")
    records, exclusions, seen = [], Counter(), set()
    for row in document["records"]:
        reason = validate_record(row, data_kind=kind)
        if reason:
            exclusions[reason] += 1
        elif row["sample_id"] in seen:
            exclusions["sample_duplicado"] += 1
        else:
            seen.add(row["sample_id"])
            records.append(row)
    return {"records": records, "excluded_counts": dict(exclusions), "manifest_sha256": hashlib.sha256(raw).hexdigest(), "data_kind": kind,
            "provenance_scope": "metadata_and_operator_attestation; source immutability must be verified upstream"}


def clinical_input_hash(row):
    """Group exact normalized pre-decision clinical inputs without their labels."""
    import unicodedata
    values = []
    for field in ("procedimiento_sistema", "hallazgos_conclusion", "descripcion_estudio_013"):
        text = unicodedata.normalize("NFKD", str(row.get(field) or ""))
        text = "".join(char for char in text if not unicodedata.combining(char))
        values.append(" ".join(text.casefold().split()))
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest() if any(values) else None


def freeze_temporal_split(records: list[dict], validation_size: float = .15, test_size: float = .15) -> dict:
    """Connect shared identities/hashes transitively; purge boundary-spanning groups."""
    if not 0 < validation_size < 1 or not 0 < test_size < 1 or validation_size + test_size >= 1:
        raise SnapshotError("fracciones_invalidas")
    if len(records) < 7:
        raise SnapshotError("muestras_insuficientes_para_tres_particiones")
    parent = list(range(len(records)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    def union(left, right):
        parent[find(right)] = find(left)
    seen = {}
    times = [timestamp(row["prediction_at"]) for row in records]
    for index, row in enumerate(records):
        keys = [("patient", row["patient_group"]), ("admission", row["admission_group"])]
        digest = (row.get("evidencia_tarifario") or {}).get("documento_sha256")
        if digest:
            keys.append(("document", digest.lower()))
        clinical_digest = clinical_input_hash(row)
        if clinical_digest:
            keys.append(("clinical_text", clinical_digest))
        for key in keys:
            if key in seen:
                union(index, seen[key])
            seen[key] = index
    components = {}
    for index in range(len(records)):
        components.setdefault(find(index), []).append(index)
    ordered = sorted(times)
    train_position = max(1, min(len(records) - 2, int(len(records) * (1 - validation_size - test_size))))
    test_position = max(train_position + 1, min(len(records) - 1, int(len(records) * (1 - test_size))))
    train_end, validation_end = ordered[train_position - 1], ordered[test_position - 1]
    partitions = {"train": [], "validation": [], "test": []}
    excluded = []
    for indices in components.values():
        earliest, latest = min(times[i] for i in indices), max(times[i] for i in indices)
        if latest <= train_end:
            name = "train"
        elif earliest > train_end and latest <= validation_end:
            name = "validation"
        elif earliest > validation_end:
            name = "test"
        else:
            excluded.extend(indices)
            continue
        partitions[name].extend(indices)
    if any(not indices for indices in partitions.values()):
        raise SnapshotError("particion_vacia_tras_purgar_grupos")
    for indices in partitions.values():
        indices.sort(key=lambda index: (times[index], records[index]["sample_id"]))
    canonical = [{"sample_id": row["sample_id"], "patient_group": row["patient_group"], "admission_group": row["admission_group"],
                  "prediction_at": row["prediction_at"], "clinical_text_sha256": clinical_input_hash(row),
                  "documento_sha256": (row.get("evidencia_tarifario") or {}).get("documento_sha256")} for row in records]
    identity = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**partitions, "excluded": sorted(excluded), "excluded_counts": {"grupo_cruza_corte_temporal": len(excluded)},
            "components": [indices for indices in components.values()], "record_identity_sha256": identity,
            "cutoffs": {"train_end": train_end.isoformat(), "validation_end": validation_end.isoformat()},
            "policy": "connected_patient_admission_document_hash_clinical_text_hash; temporal_holdout; purge_crossing_components"}


# INICIO CAMBIO AUDITORAI TARIFARIO: manifiesto común con procedencia reproducible del pipeline.
def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shared_pipeline_metadata(root: str | Path) -> dict:
    root = Path(root).resolve()
    sources = {
        "feature_builder": root / "app" / "features.py",
        "split_builder": root / "scripts" / "tariff_snapshot_dataset.py",
        "tfidf_trainer": root / "scripts" / "train_tariff_candidate.py",
        "transformer_trainer": root / "scripts" / "train_transformer_tariff_candidate.py",
    }
    if any(not path.is_file() for path in sources.values()):
        raise SnapshotError("fuentes_del_pipeline_incompletas")
    packages = {}
    for name in ("numpy", "scikit-learn", "joblib", "torch", "transformers"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    from app.features import FEATURE_SCHEMA_VERSION

    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "source_sha256": {name: _source_sha256(path) for name, path in sources.items()},
        "split_arguments": {"validation_size": 0.15, "test_size": 0.15},
        "environment": {
            "python": platform.python_version(),
            "implementation": sys.implementation.name,
            "platform": platform.platform(),
            "packages": packages,
        },
    }


def resolve_split_manifest(
    records: list[dict],
    dataset_sha256: str,
    pipeline_metadata: dict,
    manifest_path: str | Path | None = None,
) -> dict:
    generated = freeze_temporal_split(records)
    if manifest_path:
        try:
            supplied = json.loads(Path(manifest_path).read_bytes())
        except (OSError, ValueError, UnicodeDecodeError) as error:
            raise SnapshotError("manifiesto_invalido") from error
        if (
            not isinstance(supplied, dict)
            or supplied.get("manifest_version") != "tariff_split_manifest_v1"
            or supplied.get("dataset_sha256") != dataset_sha256
        ):
            raise SnapshotError("manifiesto_no_corresponde_al_dataset_o_politica")
        if any(supplied.get(key) != value for key, value in generated.items()):
            raise SnapshotError("manifiesto_no_corresponde_al_dataset_o_politica")
        supplied_pipeline = supplied.get("pipeline") or {}
        if (
            supplied_pipeline.get("feature_schema_version")
            != pipeline_metadata.get("feature_schema_version")
            or supplied_pipeline.get("source_sha256") != pipeline_metadata.get("source_sha256")
            or supplied_pipeline.get("split_arguments") != pipeline_metadata.get("split_arguments")
        ):
            raise SnapshotError("manifiesto_no_corresponde_al_dataset_o_politica")
        return supplied
    return {
        **generated,
        "manifest_version": "tariff_split_manifest_v1",
        "dataset_sha256": dataset_sha256,
        "pipeline": pipeline_metadata,
    }
# FIN CAMBIO AUDITORAI TARIFARIO: manifiesto común con procedencia reproducible del pipeline.
# FIN CAMBIO AUDITORAI TARIFARIO: snapshots previos y particiones sin fuga.
