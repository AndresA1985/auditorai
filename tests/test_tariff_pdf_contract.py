"""Synthetic GEN/AuditorAI contract; no DB, model loading, network or training."""
import ast
import asyncio
import json
from copy import deepcopy
import io
import logging
from pathlib import Path
import unittest

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from app.schemas import PrediccionPayload, PrediccionRequest, PrediccionResponse
from app.tariff_evidence import annotate_documentary_evidence, DECISION_FIELDS


def evidence(codes=("43271",), *, origin="texto"):
    return {
        "estado": "extraido" if codes else "sin_codigos",
        "fuente": origin if codes else "ninguna",
        "documento_sha256": "a" * 64,
        "codigos": [{"codigo": code, "descripcion": "Procedimiento sintético QA",
            "origen": origin, "pagina": 1,
            "confianza_ocr": 81.5 if origin == "ocr" else None} for code in codes],
        "advertencias": ["requiere_verificacion_ocr"] if origin == "ocr" else [],
    }


def prediction(codes=("43271",)):
    return {"codigos": list(codes), "codigo_scores": {"43271": .123456789},
        "codigo_ranking": [{"codigo": "43271", "score": .123456789, "selected": "43271" in codes},
                           {"codigo": "43264", "score": .112233445, "selected": "43264" in codes}],
        "honorarios_codigo": {"43271": "75"}, "honorario": "43271(75)",
        "tiempo_anestesia": "2", "tiempos_anestesia_codigo": {"43271": "2"},
        "nombre_procedimiento": "Sintético QA", "observacion_auditor": "Revisar."}


def request(document=None, **fields):
    return PrediccionRequest(id_agenda=1, evidencia_tarifario=document, **fields)


def isolated_app(predict):
    # Compile actual routes without importing DB/config/predictor or model runtimes.
    path = Path(__file__).resolve().parents[1] / "app" / "main_base.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    namespace = {"FastAPI": FastAPI, "HTTPException": HTTPException,
        "PrediccionRequest": PrediccionRequest, "PrediccionResponse": PrediccionResponse,
        "predecir": predict}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["app"]


def post_json(app, payload):
    """Minimal ASGI harness; no optional HTTP client or external socket required."""
    messages = []
    async def run():
        async def receive():
            return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}
        async def send(message):
            messages.append(message)
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": "/predecir_auditoria",
            "raw_path": b"/predecir_auditoria", "query_string": b"",
            "root_path": "", "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1), "server": ("synthetic-test", 80)}
        await app(scope, receive, send)
    asyncio.run(run())
    status = next(item["status"] for item in messages if item["type"] == "http.response.start")
    body = b"".join(item.get("body", b"") for item in messages if item["type"] == "http.response.body")
    return status, json.loads(body)


class TariffPdfContractTests(unittest.TestCase):
    def test_every_gen_state_roundtrips_exactly_through_prediccion_request(self):
        cases = [evidence(), evidence(origin="ocr"), evidence(())]
        cases += [{"estado": state, "fuente": "ninguna", "codigos": [],
                   "documento_sha256": "b" * 64 if state == "ilegible" else None,
                   "advertencias": []} for state in ("ilegible", "sin_documento", "sin_archivo", "sin_pdf")]
        for case in cases:
            with self.subTest(state=case["estado"], source=case["fuente"]):
                self.assertEqual(request(case).evidencia_tarifario.model_dump(), case)

    def test_raw_pdf_ocr_and_unknown_nested_fields_are_rejected(self):
        for key in ("pdf", "pdf_base64", "ocr_completo", "texto_completo", "token"):
            with self.subTest(key=key):
                case = evidence(); case[key] = "SYNTHETIC_PRIVATE_SENTINEL"
                with self.assertRaises(ValidationError): request(case)
                case = evidence(); case["codigos"][0][key] = "SYNTHETIC_PRIVATE_SENTINEL"
                with self.assertRaises(ValidationError): request(case)

    def test_eight_digit_references_preserve_strict_documentary_boundary(self):
        for origin in ("texto", "ocr"):
            case = evidence(("70200003", "70200004"), origin=origin)
            case["advertencias"].extend(["codigo_validacion_no_verificable", "formato_no_identificado"])
            validated = request(case)
            self.assertEqual([item.codigo for item in validated.evidencia_tarifario.codigos], ["70200003", "70200004"])
            result = annotate_documentary_evidence(validated, prediction())
            for field in DECISION_FIELDS:
                self.assertEqual(result.get(field), prediction().get(field))
            self.assertEqual([item["codigo"] for item in result["opciones_tarifario_documental"]], ["70200003", "70200004"])
            PrediccionPayload(**result)
            for nested in (False, True):
                invalid = deepcopy(case)
                target = invalid["codigos"][0] if nested else invalid
                target["codigo_validacion"] = "CV_SYNTHETIC_LOCAL_ONLY"
                with self.assertRaises(ValidationError):
                    request(invalid)

    def test_eight_digit_clinical_contradictions_keep_each_original_source(self):
        for field in ("hallazgos_conclusion", "descripcion_estudio_013", "informe_tecnico_justificacion"):
            quote = "No se realizó 70200003 en la prueba sintética."
            original = prediction()
            result = annotate_documentary_evidence(request(evidence(("70200003",)), **{field: quote}), original)
            self.assertTrue(any(item["codigo"] == "70200003" and item["tipo"] == "contradiccion_clinica" for item in result["discrepancias"]))
            self.assertTrue(any(item["campo"] == field and item["texto"] in quote and item["verificada"]
                                for item in result["citas_documentales"] if item["alcance"] == "clinica_literal"))
            for decision in DECISION_FIELDS:
                self.assertEqual(result.get(decision), original.get(decision))

    def test_structured_evidence_enforces_bounded_counts_page_description_and_hash(self):
        mutations = [lambda x: x.update(documento_sha256=None),
            lambda x: x.update(documento_sha256="invalid"),
            lambda x: x.update(codigos=x["codigos"] * 26),
            lambda x: x.update(advertencias=["requiere_verificacion_ocr"] * 26),
            lambda x: x["codigos"][0].update(pagina=11),
            lambda x: x["codigos"][0].update(pagina=True),
            lambda x: x["codigos"][0].update(descripcion="a" * 181),
            lambda x: x["codigos"][0].update(codigo="1234567890"),
            lambda x: x["codigos"][0].update(codigo="CV43271"),
            lambda x: x["codigos"][0].update(confianza_ocr=99),
            lambda x: x.update(estado="sin_codigos"),
            lambda x: x.update(fuente="ocr")]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                case = evidence(); mutate(case)
                with self.assertRaises(ValidationError): request(case)

    def test_each_document_code_is_separate_high_priority_without_automatic_selection(self):
        original = prediction(); before = deepcopy(original)
        result = annotate_documentary_evidence(request(evidence(("43271", "43264", "45380"))), original)
        self.assertEqual(original, before)
        for field in DECISION_FIELDS: self.assertEqual(result.get(field), before.get(field))
        options = result["opciones_tarifario_documental"]
        self.assertEqual([item["codigo"] for item in options], ["43271", "43264", "45380"])
        self.assertTrue(all(item["prioridad"] == "alta" and item["requiere_revision"] for item in options))
        self.assertEqual([item["modelo_selected"] for item in options], [True, False, False])
        self.assertEqual([item["modelo_score"] for item in options], [.123456789, .112233445, None])
        PrediccionPayload(**result)

    def test_native_and_ocr_agreement_and_divergence_preserve_all_decisions(self):
        for origin in ("texto", "ocr"):
            for codes in (("43271",), ("45380",)):
                with self.subTest(origin=origin, codes=codes):
                    original = prediction(); source = request(evidence(codes, origin=origin))
                    result = annotate_documentary_evidence(source, original)
                    for field in DECISION_FIELDS: self.assertEqual(result.get(field), original.get(field))
                    self.assertEqual([item["score_ranking"] for item in result["comparacion_ranking_tarifario"]], [.123456789, .112233445])
                    self.assertEqual([item["selected"] for item in result["comparacion_ranking_tarifario"]], [True, False])
                    self.assertTrue(result["requiere_revision"])
                    self.assertEqual(bool(result["discrepancias"]), codes != ("43271",))
                    if origin == "ocr":
                        self.assertEqual(result["opciones_tarifario_documental"][0]["confianza_ocr"], 81.5)
                    PrediccionPayload(**result)

    def test_genuinely_absent_pdf_preserves_legacy_clinical_response_exactly(self):
        for case in (None, {"estado": "sin_documento", "fuente": "ninguna", "codigos": [], "advertencias": ["sin_documento"]},
                     {"estado": "sin_archivo", "fuente": "ninguna", "codigos": [], "advertencias": ["sin_archivo"]}):
            original = prediction()
            self.assertEqual(annotate_documentary_evidence(request(case), original), original)

    def test_missing_expected_pdf_requires_review_without_changing_clinical_response(self):
        for state in ("sin_documento", "sin_pdf"):
            case = {"estado": state, "fuente": "ninguna", "codigos": [], "advertencias": ["pdf_no_adjuntado"]}
            original = prediction(); result = annotate_documentary_evidence(request(case), original)
            self.assertTrue(result["requiere_revision"])
            self.assertFalse(result["abstencion"])
            for field in DECISION_FIELDS: self.assertEqual(result.get(field), original.get(field))

    def test_empty_model_output_always_abstains_even_when_pdf_has_codes(self):
        for case in (None, evidence(), evidence(())):
            original = prediction(()); result = annotate_documentary_evidence(request(case), original)
            self.assertEqual(result["codigos"], [])
            self.assertTrue(result["abstencion"])
            self.assertTrue(result["requiere_revision"])
            self.assertEqual(result["motivo_abstencion"], "sin_codigos_sobre_umbral")
            for field in DECISION_FIELDS: self.assertEqual(result.get(field), original.get(field))
            PrediccionPayload(**result)

    def test_clinical_contradiction_in_each_field_requires_review_and_correct_citation(self):
        for field in ("hallazgos_conclusion", "descripcion_estudio_013", "informe_tecnico_justificacion"):
            quote = "No se realizó 43271 en la prueba sintética."
            result = annotate_documentary_evidence(request(evidence(), **{field: quote}), prediction())
            self.assertTrue(result["requiere_revision"])
            self.assertTrue(any(item["tipo"] == "contradiccion_clinica" for item in result["discrepancias"]))
            citations = [item for item in result["citas_documentales"] if item["alcance"] == "clinica_literal"]
            self.assertTrue(any(item["campo"] == field and item["texto"] in quote for item in citations))

    def test_real_endpoint_routes_use_contract_with_stubbed_prediction(self):
        seen = []
        def predict(req):
            seen.append(req)
            return annotate_documentary_evidence(req, prediction()), .123456789
        status, response = post_json(isolated_app(predict), {"id_agenda": 1, "evidencia_tarifario": evidence()})
        self.assertEqual(status, 200)
        self.assertIsInstance(seen[0], PrediccionRequest)
        body = response["prediccion"]
        self.assertEqual(body["codigo_ranking"][0]["score"], .123456789)
        self.assertTrue(body["codigo_ranking"][0]["selected"])

    def test_internal_exception_text_never_leaves_endpoint_or_appears_in_application_logs(self):
        marker = "SYNTHETIC_PRIVATE_SENTINEL_DO_NOT_LOG"
        for error, expected in ((ValueError(marker), 422), (RuntimeError(marker), 500)):
            def failing(_): raise error
            stream = io.StringIO(); handler = logging.StreamHandler(stream)
            logging.getLogger().addHandler(handler)
            try:
                status, response = post_json(isolated_app(failing), {"id_agenda": 1, "hallazgos_conclusion": marker})
                self.assertEqual(status, expected)
                self.assertNotIn(marker, json.dumps(response))
                self.assertNotIn(marker, stream.getvalue())
            finally:
                logging.getLogger().removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
