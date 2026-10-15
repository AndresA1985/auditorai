import os
from pathlib import Path

ENV_DEFAULTS = {
    "DB_HOST": "localhost", "DB_PORT": "3306", "DB_DATABASE": "test",
    "DB_USERNAME": "test", "DB_PASSWORD": "", "AUDITORIA_MIN_SCORE": "0.1",
    "AUDITORIA_MODEL_PATH": "models/test.joblib", "AUDITORIA_MODEL_MIN_SIMILARITY": "0.1",
    "AUDITORIA_SCORE_EXACT_MATCH_BONUS": "0.1", "AUDITORIA_SCORE_TOKEN_MATCH_BONUS": "0.1",
    "AUDITORIA_SCORE_CODE_GROUP_BONUS": "0.1", "AUDITORIA_VALIDATION_SIZE": "0.15",
    "AUDITORIA_TEST_SIZE": "0.15", "AUDITORIA_MODEL_ENGINE": "tfidf",
    "AUDITORIA_EMBEDDING_MODEL": "test", "AUDITORIA_EMBEDDING_DEVICE": "cpu",
    "AUDITORIA_EMBEDDING_BATCH_SIZE": "8",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

from app import ml_model
from app.ml_model import anexar_justificaciones_codigo
from app.model_consensus import reconcile_autoplan_candidates
from app.schemas import AutoplanCandidates, PrediccionPayload, PrediccionRequest


FINDING = (
    "BAJO ANESTESIA GENERAL SE INTRODUCE EL ECOENDOSCOPIO LINEAL. "
    "CON AGUJA DE 22 FNB SE PROCEDE A REALIZAR TOMA DE BIOPSIA HEPATICA."
)


def candidates(*items):
    return AutoplanCandidates.model_validate({
        "model": "tfidf",
        "artifact_sha256": "a" * 64,
        "threshold": 0.2,
        "abstained": not items,
        "candidate_codes": list(items),
        "publication_allowed": False,
    })


def prediction():
    return {
        "codigos": ["43259"],
        "codigo_scores": {"43259": 0.209093},
        "codigo_ranking": [
            {"codigo": "43259", "score": 0.209093, "selected": True},
            {"codigo": "43242", "score": 0.050237, "selected": False},
        ],
        "honorarios_codigo": {},
        "honorario": "",
        "tiempo_anestesia": "",
        "nombre_procedimiento": "ECOENDOSCOPIA DIAGNOSTICA",
        "observacion_auditor": "Propuesta original.",
    }


def test_autoplan_candidate_replaces_generic_eus_and_preserves_original_score():
    evidence = candidates({
        "code": "43242",
        "score": 0.9705,
        "support_text": "CON AGUJA DE 22 FNB SE PROCEDE A REALIZAR TOMA DE BIOPSIA HEPATICA.",
    })
    result, score, supports = reconcile_autoplan_candidates(prediction(), evidence)
    assert result["codigos"] == ["43242"]
    assert score == 0.9705
    assert result["codigo_scores"] == {"43242": 0.9705}
    assert [(item["codigo"], item["selected"]) for item in result["codigo_ranking"]] == [
        ("43242", True), ("43259", False)
    ]
    chosen = result["codigo_ranking"][0]
    assert chosen["score"] == 0.9705
    assert chosen["score_auditorai_original"] == 0.050237
    assert chosen["fuente_score"] == "consenso_autoplan"
    assert supports["43242"] in FINDING
    assert result["consenso_modelos"]["codigos_auditorai_originales"] == ["43259"]
    assert result["requiere_revision"] is True


def test_abstained_autoplan_does_not_mutate_auditorai():
    original = prediction()
    result, score, supports = reconcile_autoplan_candidates(original, candidates())
    assert result == original
    assert score == 0.0 and supports == {}


def test_candidate_contract_never_enters_model_features_or_accepts_publication():
    request = PrediccionRequest.model_validate({
        "id_agenda": 1,
        "hallazgos_conclusion": FINDING,
        "autoplan_candidatos": {
            "model": "tfidf", "artifact_sha256": "b" * 64,
            "threshold": 0.2, "abstained": False,
            "candidate_codes": [{"code": "43242", "score": 0.97,
                                  "support_text": FINDING.split(". ")[1]}],
            "publication_allowed": False,
        },
    })
    assert request.autoplan_candidatos.candidate_codes[0].code == "43242"
    payload = prediction()
    payload["consenso_modelos"] = {"estado": "aplicado"}
    PrediccionPayload.model_validate(payload)


def test_support_ia_uses_the_same_autoplan_code_score_and_literal():
    literal = "CON AGUJA DE 22 FNB SE PROCEDE A REALIZAR TOMA DE BIOPSIA HEPATICA."
    ranking = [{
        "codigo": "43242", "score": 0.9705, "selected": True,
        "score_auditorai_original": 0.050237, "score_autoplan": 0.9705,
        "fuente_score": "consenso_autoplan",
    }]
    result = anexar_justificaciones_codigo(ranking, FINDING, {"43242": literal})
    item = result[0]
    assert item["texto_soporte"] == literal
    assert "Autoplan sugirió el código 43242" in item["justificacion"]
    assert item["evidencias"]["codigo"]["fuente"] == "consenso_autoplan"
    assert item["evidencias"]["codigo"]["score_ranking"] == 0.9705


def test_full_code_prediction_applies_consensus_before_documentary_support(monkeypatch):
    literal = "CON AGUJA DE 22 FNB SE PROCEDE A REALIZAR TOMA DE BIOPSIA HEPATICA."
    request = PrediccionRequest.model_validate({
        "id_agenda": 1,
        "procedimiento_sistema": "BIOPSIA / ECOENDOSCOPIA DIAGNOSTICA",
        "hallazgos_conclusion": FINDING,
        "autoplan_candidatos": {
            "model": "tfidf", "artifact_sha256": "e" * 64,
            "threshold": 0.2, "abstained": False,
            "candidate_codes": [{"code": "43242", "score": 0.9705,
                                  "support_text": literal}],
            "publication_allowed": False,
        },
    })
    artifact = {"model": "tfidf_logistic_regression_multilabel"}
    monkeypatch.setattr(ml_model._base, "cargar_modelo", lambda _: artifact)
    monkeypatch.setattr(ml_model._base, "texto_request", lambda *_: "texto")
    monkeypatch.setattr(ml_model._base, "es_artefacto_tarifario", lambda *_: False)
    monkeypatch.setattr(ml_model._base, "predecir_multilabel", lambda *_: (prediction(), 0.209093))
    monkeypatch.setattr(ml_model, "soportes_por_codigo", lambda *_: {"43242": "SE OBSERVA NORMAL."})
    result, score = ml_model.predecir_con_modelo(request, Path("ignored.joblib"))
    assert result["codigos"] == ["43242"] and score == 0.9705
    assert result["codigo_ranking"][0]["texto_soporte"] == literal
    assert result["codigo_ranking"][0]["selected"] is True
    assert "Autoplan sugirió el código 43242" in result["codigo_ranking"][0]["justificacion"]
