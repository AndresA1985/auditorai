# INICIO CAMBIO AUDITORIA TARIFARIO: comparación documental independiente de las decisiones del modelo.
"""Trace extracted context without inventing PDF or clinical justifications."""
from copy import deepcopy
import re
import unicodedata
from .features import as_payload, documentary_references

CLINICAL_SOURCES = ("hallazgos_conclusion", "descripcion_estudio_013", "informe_tecnico_justificacion")
DECISION_FIELDS = (
    "codigos", "codigo_scores", "codigo_ranking", "honorarios_codigo", "honorario",
    "tiempos_anestesia_codigo", "tiempo_anestesia", "nombre_procedimiento",
)
ALLOWED_DIFFERENCES = {
    "adicional_modelo", "ausente_modelo", "sin_evidencia_documental", "contradiccion_clinica"
}
NEGATION = re.compile(r"\bno\s+(?:se\s+)?(?:realiz\w*|practic\w*|efectu\w*|hizo|llev\w*\s+a\s+cabo)\b", re.I)
GENERIC_WORDS = {"codigo", "tarifario", "procedimiento", "observaciones", "auditoria", "para", "con", "sin", "del", "paciente"}


def _fold(text):
    return "".join(char for char in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(char) != "Mn")


def _explicit_negation(hallazgo: str, reference: dict) -> str | None:
    """A narrow textual warning, never a diagnosis or a reason to delete a code."""
    words = {token for token in re.findall(r"[a-z0-9]+", _fold(str(reference.get("descripcion") or "")))
        if len(token) >= 4 and token not in GENERIC_WORDS}
    for match in NEGATION.finditer(hallazgo):
        clause = re.split(r"[.!?;\r\n]", hallazgo[match.start():match.end() + 120], maxsplit=1)[0].strip()
        normalized = _fold(clause)
        if re.search(r"(?<!\w)" + re.escape(reference["codigo"]) + r"(?!\w)", normalized) or words & set(re.findall(r"[a-z0-9]+", normalized)):
            return clause
    return None


def _decision_state(prediction: dict) -> dict:
    return {field: deepcopy(prediction.get(field)) for field in DECISION_FIELDS}


def _return_with_intact_decisions(result: dict, original_state: dict) -> dict:
    if _decision_state(result) != original_state:
        raise RuntimeError("La evidencia documental no puede mutar decisiones del modelo")
    return result


def annotate_documentary_evidence(req, prediction: dict, *, candidate: bool = False) -> dict:
    payload = as_payload(req)
    result = deepcopy(prediction)
    original_state = _decision_state(result)
    previous_review = bool(result.get("requiere_revision", False))
    previous_differences = deepcopy(result.get("discrepancias") or [])
    if not isinstance(previous_differences, list): raise ValueError("Invalid previous discrepancies")
    if any(
        not isinstance(item, dict)
        or not re.fullmatch(r"[0-9]{5,8}", str(item.get("codigo") or ""))
        or item.get("tipo") not in ALLOWED_DIFFERENCES
        or not isinstance(item.get("motivo"), str)
        or not 1 <= len(item["motivo"]) <= 500
        for item in previous_differences
    ):
        raise ValueError("Invalid previous discrepancies")
    evidence = deepcopy(payload.get("evidencia_tarifario"))
    absent = evidence is None or (evidence.get("estado") in {"sin_documento", "sin_archivo", "sin_pdf"} and not evidence.get("codigos"))
    missing_expected_pdf = bool(evidence and (
        evidence.get("estado") == "sin_pdf"
        or "pdf_no_adjuntado" in evidence.get("advertencias", [])
    ))
    if absent and not missing_expected_pdf and not candidate and result.get("codigos"):
        return _return_with_intact_decisions(result, original_state)
    if evidence is None:
        evidence = {"estado": "sin_documento", "fuente": "ninguna", "codigos": [], "advertencias": ["sin_documento"]}
    if absent:
        # No authorization document is a normal clinical-only case, not an error.
        abstained = not bool(result.get("codigos"))
        reason = result.get("motivo_abstencion") or (
            "sin_codigos_sobre_umbral" if abstained else None
        )
        # INICIO CAMBIO AUDITORAI TARIFARIO: ausencia conserva el ranking serializable completo.
        ranking_comparison = [
            {
                "codigo": item["codigo"],
                "score_ranking": item["score"],
                "selected": bool(item.get("selected")),
                "en_documento": False,
                "clasificacion": "sin_referencia_documental",
                "citas": [],
                "semantica_score": (
                    "Puntaje original de ranking no calibrado del modelo; independiente "
                    "de confianza OCR y de autorización documental."
                ),
            }
            for item in result.get("codigo_ranking") or []
        ]
        # FIN CAMBIO AUDITORAI TARIFARIO: ausencia conserva el ranking serializable completo.
        result.update({"evidencia_tarifario": evidence, "requiere_revision": bool(previous_review or previous_differences or abstained or missing_expected_pdf),
            "discrepancias": previous_differences, "citas_documentales": [], "justificaciones_tarifario": [],
            "comparacion_ranking_tarifario": ranking_comparison, "opciones_tarifario_documental": [],
            "abstencion": abstained,
            "motivo_abstencion": reason})
        return _return_with_intact_decisions(result, original_state)
    references = documentary_references({"evidencia_tarifario": evidence})
    selected = set(result.get("codigos") or [])
    documentary = {item["codigo"] for item in references}
    differences = previous_differences
    for code in sorted(selected - documentary):
        differences.append({"codigo": code, "tipo": "adicional_modelo" if documentary else "sin_evidencia_documental",
            "motivo": "La selección no coincide con una referencia documental; requiere contraste clínico del auditor. No confirma una complicación."})
    for code in sorted(documentary - selected):
        differences.append({"codigo": code, "tipo": "ausente_modelo",
            "motivo": "La referencia documental no está seleccionada; su presencia no demuestra que el procedimiento se realizó."})
    citations = []
    for index, item in enumerate(references):
        for field in ("codigo", "descripcion"):
            quote = str(item.get(field) or "")
            if not quote: continue
            citations.append({"codigo": item["codigo"], "fuente": "documento_ocr" if item["origen"] == "ocr" else "documento_texto",
                "campo": "evidencia_tarifario.codigos[%d].%s" % (index, field), "texto": quote,
                "pagina": item["pagina"], "documento_sha256": evidence.get("documento_sha256"),
                # El hash y la estructura preservan procedencia declarada, pero este servicio
                # no vuelve a abrir el PDF y por eso no certifica la cita contra sus bytes.
                "verificada": False, "alcance": "contexto_extraido_gen"})
    # Check every declared clinical source; preserve the literal and its own field.
    for reference in references:
        for field in CLINICAL_SOURCES:
            quote = _explicit_negation(str(payload.get(field) or ""), reference)
            if quote:
                differences.append({"codigo": reference["codigo"], "tipo": "contradiccion_clinica",
                    "motivo": "Posible contradicción textual: una negación explícita coincide con el término documental. No es una conclusión clínica; requiere revisión del auditor."})
                citations.append({"codigo": reference["codigo"], "fuente": field, "campo": field,
                    "texto": quote, "verificada": True, "alcance": "clinica_literal"})
    # A model's quotation is accepted only as a literal in its own clinical source.
    for item in result.get("codigo_ranking") or []:
        if item.get("codigo") not in selected: continue
        quote = item.get("texto_soporte")
        if not isinstance(quote, str) or not quote or len(quote) > 500: continue
        for field in CLINICAL_SOURCES:
            if quote in str(payload.get(field) or ""):
                citations.append({"codigo": item["codigo"], "fuente": field, "campo": field,
                    "texto": quote, "verificada": True, "alcance": "clinica_literal"})
                break
    # Prior clinical conclusions keep their source-specific reason; add each difference once.
    merged = []
    seen_differences = set()
    for difference in differences:
        key = (difference["codigo"], difference["tipo"])
        if key in seen_differences: continue
        seen_differences.add(key); merged.append(difference)
    differences = merged
    clinical_codes = {item["codigo"] for item in citations if item["alcance"] == "clinica_literal"}
    # JSON estructurado no equivale a verificación causal ni visual del PDF.
    review = bool(
        previous_review
        or references
        or differences
        or selected - clinical_codes
        or evidence.get("advertencias")
        or evidence.get("estado") != "extraido"
        or any(item.get("origen") == "ocr" for item in references)
    )
    explanations = []
    for code in sorted(selected | documentary):
        related = [citation for citation in citations if citation["codigo"] == code]
        clinical = any(item["alcance"] == "clinica_literal" for item in related)
        explanations.append({"codigo": code, "en_documento": code in documentary, "seleccionado": code in selected,
            "citas": related, "requiere_revision": review or not clinical,
            "justificacion": "La referencia documental es contexto de autorización. La verificación de una cita comprueba su literalidad, no demuestra sustento clínico ni realización; la decisión corresponde al auditor."})
    ranking_comparison = []
    for item in result.get("codigo_ranking") or []:
        code = item["codigo"]
        in_document = code in documentary
        classification = ("coincidencia" if code in selected else "documental_no_seleccionado") if in_document else ("adicional" if documentary else "sin_referencia_documental")
        ranking_comparison.append({"codigo": code, "score_ranking": item["score"], "selected": bool(item.get("selected")),
            "en_documento": in_document, "clasificacion": classification,
            "citas": [citation for citation in citations if citation["codigo"] == code and citation["alcance"] == "contexto_extraido_gen"],
            "semantica_score": "Puntaje original de ranking no calibrado del modelo; independiente de confianza OCR y de autorización documental."})
    ranked = {item["codigo"]: item for item in result.get("codigo_ranking") or []}
    contradicted = {item["codigo"] for item in differences if item["tipo"] == "contradiccion_clinica"}
    options = []
    seen = set()
    for reference in references:
        code = reference["codigo"]
        if code in seen: continue
        seen.add(code)
        reason = ("Posible contradicción por negación explícita: revisar el documento y los hallazgos."
            if code in contradicted else "Tarifario explícito extraído del documento, presentado como referencia principal; no modifica la selección del modelo.")
        if reference["origen"] == "ocr": reason += " El OCR requiere verificación visual."
        options.append({"codigo": code, "descripcion": reference.get("descripcion") or "",
            "prioridad": "alta", "requiere_revision": True,
            "etiqueta": "Referencia principal del documento", "fuente": "documento_codigo_validacion",
            "origen": reference["origen"], "pagina": reference["pagina"], "documento_sha256": evidence.get("documento_sha256"),
            "confianza_ocr": reference.get("confianza_ocr"), "modelo_score": ranked[code]["score"] if code in ranked else None,
            "modelo_selected": code in selected, "coincidencia_modelo": code in ranked,
            "citas": [citation for citation in citations if citation["codigo"] == code], "motivo": reason})
    result.update({"evidencia_tarifario": evidence, "requiere_revision": review,
        "discrepancias": differences, "citas_documentales": citations, "justificaciones_tarifario": explanations,
        "abstencion": not bool(selected), "comparacion_ranking_tarifario": ranking_comparison,
        "opciones_tarifario_documental": options})
    if not selected:
        result["motivo_abstencion"] = result.get("motivo_abstencion") or "sin_codigos_sobre_umbral"
    return _return_with_intact_decisions(result, original_state)
# FIN CAMBIO AUDITORIA TARIFARIO: comparación documental independiente de las decisiones del modelo.
