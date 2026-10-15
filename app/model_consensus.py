"""Reconcile AuditorAI ranking with Autoplan's read-only candidate model."""
from __future__ import annotations

from copy import deepcopy


def reconcile_autoplan_candidates(prediction: dict, candidates) -> tuple[dict, float, dict[str, str]]:
    """Use Autoplan's reviewed candidate ranking while retaining AuditorAI traceability.

    The candidate service is not a publisher.  Its output only changes the
    suggestion shown to the auditor and always leaves the case in review.
    """
    result = deepcopy(prediction)
    if candidates is None or candidates.abstained or not candidates.candidate_codes:
        return result, 0.0, {}

    original_codes = list(result.get("codigos") or [])
    original_ranking = deepcopy(result.get("codigo_ranking") or [])
    originals = {
        str(item.get("codigo") or ""): item
        for item in original_ranking
        if isinstance(item, dict) and item.get("codigo")
    }
    selected_codes = [item.code for item in candidates.candidate_codes]
    selected = set(selected_codes)
    supports = {
        item.code: item.support_text
        for item in candidates.candidate_codes
        if item.support_text
    }

    ranking = []
    for candidate in candidates.candidate_codes:
        item = deepcopy(originals.get(candidate.code) or {"codigo": candidate.code})
        original_score = item.get("score")
        item.update({
            "score": round(float(candidate.score), 8),
            "selected": True,
            "score_auditorai_original": original_score,
            "score_autoplan": round(float(candidate.score), 8),
            "fuente_score": "consenso_autoplan",
        })
        ranking.append(item)
    for item in original_ranking:
        code = str(item.get("codigo") or "")
        if not code or code in selected:
            continue
        copied = deepcopy(item)
        copied.update({
            "selected": False,
            "score_auditorai_original": copied.get("score"),
            "score_autoplan": None,
            "fuente_score": "auditorai",
        })
        ranking.append(copied)

    result["codigos"] = selected_codes
    result["codigo_scores"] = {
        item.code: round(float(item.score), 8) for item in candidates.candidate_codes
    }
    result["codigo_ranking"] = ranking
    result["requiere_revision"] = True
    result["abstencion"] = False
    result.pop("motivo_abstencion", None)
    result["consenso_modelos"] = {
        "estado": "aplicado",
        "fuente_decision": "autoplan_candidate",
        "modelo_autoplan": candidates.model,
        "artifact_sha256": candidates.artifact_sha256,
        "threshold": candidates.threshold,
        "publication_allowed": False,
        "codigos_auditorai_originales": original_codes,
        "codigos_autoplan": selected_codes,
        "requiere_revision": True,
    }
    result["observacion_auditor"] = (
        "Códigos sugeridos mediante consenso con Autoplan; revise la evidencia clínica "
        "antes de guardar. Autoplan no publica planillas automáticamente."
    )
    return result, max(result["codigo_scores"].values()), supports
