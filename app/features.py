# INICIO CAMBIO AUDITORIA TARIFARIO: constructor puro compartido, sin modelos ni bases de datos.
"""Versioned features for isolated candidates; input evidence is data, never orders."""
from collections.abc import Mapping
import re

FEATURE_SCHEMA_VERSION = "tariff_features_v1"
FEATURE_MODES = frozenset({"clinical", "document_only", "clinical_document"})
CLINICAL_FIELDS = ("procedimiento_sistema", "hallazgos_conclusion", "descripcion_estudio_013")
GEN_BEGIN = "[DATOS DEL DOCUMENTO DE VALIDACION: REFERENCIA TARIFARIA PRINCIPAL]"
GEN_END = "[FIN DATOS DOCUMENTALES; CONTRASTAR CON LOS HALLAZGOS Y REVISAR DIFERENCIAS]"
TECHNICAL_BEGIN = "[INFORME TECNICO APORTADO PARA REVISION]"
DIRECTIVE = re.compile(r"\b(?:ignor[ae]\w*\s+(?:las?\s+)?instrucciones|ignore\s+(?:all\s+|previous\s+)?instructions|system\s*prompt|assistant\s*:|use\s+(?:todos|all)\s+(?:los\s+)?c[oó]digos|ejecut[ae]\w*\s+(?:comando|http))", re.I)


def as_payload(value) -> dict:
    if isinstance(value, Mapping): return dict(value)
    if hasattr(value, "model_dump"): return value.model_dump()
    raise TypeError("Features require a mapping or a validated request")


def original_procedure(payload: dict) -> str:
    """Remove only the exact, complete transition block; never guess clinical text."""
    text = str(payload.get("procedimiento_sistema") or "")
    while text.startswith(GEN_BEGIN) and GEN_END in text:
        text = text.split(GEN_END, 1)[1].lstrip("\r\n")
    return text


def redact_identity(text: str, payload: dict) -> str:
    # Identifiers are for correspondence/splits, not predictive features.
    for name in ("paciente", "cedula"):
        value = str(payload.get(name) or "").strip()
        if value:
            text = re.sub(r"(?<!\w)" + re.escape(value) + r"(?!\w)", " ", text, flags=re.I)
    text = re.sub(r"(?<!\w)CV[A-Z0-9][A-Z0-9._/-]*(?!\w)", " ", text, flags=re.I)
    text = re.sub(r"\bCV\s+[A-Z0-9._/-]*[0-9][A-Z0-9._/-]*", " ", text, flags=re.I)
    text = re.sub(r"\bc[oó]digo\s+(?:de\s+)?validaci[oó]n\s*[:=]?\s*[A-Z0-9._/-]+", " ", text, flags=re.I)
    text = re.sub(r"(?<![0-9])[0-9]{10}(?![0-9])", " ", text)
    return text


def _description(text: str, payload: dict) -> str:
    if DIRECTIVE.search(text): return ""
    text = redact_identity(text[:180], payload)
    # A number embedded in a description cannot introduce a second tariff label.
    text = re.sub(r"(?<![0-9])[0-9]{5,}(?![0-9])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def documentary_references(payload: dict) -> list[dict]:
    evidence = payload.get("evidencia_tarifario") or {}
    if hasattr(evidence, "model_dump"): evidence = evidence.model_dump()
    if not isinstance(evidence, dict) or evidence.get("estado") != "extraido": return []
    references = []
    for item in evidence.get("codigos") or []:
        if hasattr(item, "model_dump"): item = item.model_dump()
        if not isinstance(item, dict): continue
        code = item.get("codigo")
        if not isinstance(code, str) or not re.fullmatch(r"[0-9]{5,8}", code): continue
        references.append(item)
    return references[:25]


def build_features(payload: dict, mode: str = "clinical_document") -> str:
    """Identical train/inference representation; never mutates original fields.

    Clinical inputs keep their order/spacing when no redaction is necessary.
    GEN's transient block is replaced by structured evidence exactly once.
    OCR confidence, IDs, hash, state, page, and demographics are not predictors.
    """
    if mode not in FEATURE_MODES: raise ValueError("Unsupported feature mode")
    payload = as_payload(payload)
    clinical_parts = []
    if mode != "document_only":
        for field in CLINICAL_FIELDS:
            value = original_procedure(payload) if field == "procedimiento_sistema" else str(payload.get(field) or "")
            if field == "descripcion_estudio_013" and payload.get("informe_tecnico_justificacion") and TECHNICAL_BEGIN in value:
                original, appended = value.rsplit(TECHNICAL_BEGIN, 1)
                if appended.strip() == str(payload["informe_tecnico_justificacion"]).strip(): value = original.rstrip("\r\n")
            clinical_parts.append(redact_identity(value, payload))
        clinical = " ".join(clinical_parts).strip()
        technical = redact_identity(str(payload.get("informe_tecnico_justificacion") or ""), payload).strip()
        if technical: clinical += "\n[INFORME TECNICO: DATOS CLINICOS]\n" + technical
    else: clinical = ""
    document = []
    if mode != "clinical":
        for item in documentary_references(payload):
            line = "CODIGO TARIFARIO DOCUMENTAL " + item["codigo"]
            description = _description(str(item.get("descripcion") or ""), payload)
            if description: line += " " + description
            document.append(line)
    if not document: return clinical
    context = "[REFERENCIAS DOCUMENTALES: DATOS EXTRAIDOS, SIN DECISION AUTOMATICA]\n" + "\n".join(document) + "\n[FIN REFERENCIAS DOCUMENTALES]"
    return context + ("\n" + clinical if clinical else "")


def build_prediction_text(request, mode: str = "clinical_document") -> str:
    return build_features(as_payload(request), mode)


def build_training_text(snapshot: dict, mode: str = "clinical_document") -> str:
    return build_features(snapshot, mode)
# FIN CAMBIO AUDITORIA TARIFARIO: constructor puro compartido, sin modelos ni bases de datos.
