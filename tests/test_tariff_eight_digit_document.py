"""Eight-digit documentary references do not alter clinical model decisions."""
from copy import deepcopy
import unittest

from pydantic import ValidationError

from app.features import build_features, documentary_references
from app.schemas import EvidenciaTarifario, OpcionTarifarioDocumental, PrediccionRequest
from app.tariff_evidence import DECISION_FIELDS, annotate_documentary_evidence
from scripts.tariff_snapshot_dataset import validate_record


def evidence():
    return {"estado": "extraido", "fuente": "texto", "documento_sha256": "a" * 64,
            "codigos": [{"codigo": code, "descripcion": "ENDOSCOPIA; BIOPSIA", "origen": "texto", "pagina": 1}
                        for code in ("70200003", "70200004")], "advertencias": []}


class EightDigitDocumentTests(unittest.TestCase):
    def test_contract_features_and_all_documentary_options_retain_both(self):
        request = PrediccionRequest(id_agenda=1, procedimiento_sistema="Procedimiento QA", evidencia_tarifario=evidence())
        before = {"codigos": ["43271"], "codigo_ranking": [{"codigo": "43271", "score": 0.79, "selected": True}],
                  "honorarios_codigo": {"43271": "20"}, "honorario": "20",
                  "tiempos_anestesia_codigo": {"43271": "10"}, "tiempo_anestesia": "10"}
        copied = deepcopy(before)
        result = annotate_documentary_evidence(request, before)
        for field in DECISION_FIELDS:
            self.assertEqual(result.get(field), copied.get(field))
        self.assertEqual(before, copied)
        self.assertEqual([option["codigo"] for option in result["opciones_tarifario_documental"]], ["70200003", "70200004"])
        for option in result["opciones_tarifario_documental"]:
            OpcionTarifarioDocumental(**option)
            self.assertIsNone(option["modelo_score"])
            self.assertFalse(option["modelo_selected"])
        self.assertIn("70200003", build_features(request))
        self.assertIn("70200004", build_features(request))
        self.assertNotIn("70200003", build_features(request, "clinical"))

    def test_numeric_codes_5_to_8_and_leading_zero_are_preserved(self):
        for code in ("43271", "123456", "1234567", "00012345"):
            item = evidence()
            item["codigos"][0]["codigo"] = code
            self.assertEqual(EvidenciaTarifario(**item).codigos[0].codigo, code)
        for code in ("1234", "123456789", "1234567890", "70200003/70200004", "70200 004", "CVTEST0001"):
            item = evidence()
            item["codigos"][0]["codigo"] = code
            with self.assertRaises(ValidationError):
                EvidenciaTarifario(**item)

    def test_unrecognised_format_warning_uses_existing_reviewable_empty_state(self):
        item = {"estado": "sin_codigos", "fuente": "ninguna", "documento_sha256": "a" * 64,
                "codigos": [], "advertencias": ["formato_no_identificado", "tabla_ambigua", "codigo_validacion_requiere_revision", "documento_estructura_ambigua"]}
        self.assertEqual(EvidenciaTarifario(**item).estado, "sin_codigos")
        self.assertEqual(documentary_references({"evidencia_tarifario": item}), [])

    def test_snapshot_accepts_document_codes_without_widening_training_labels(self):
        row = {"sample_id": "sample-qa", "patient_group": "patient-qa", "admission_group": "admission-qa",
               "snapshot_immutable": True, "prediction_at": "2026-01-02T00:00:00+00:00",
               "clinical_snapshot_captured_at": "2026-01-01T00:00:00+00:00",
               "clinical_available_at": "2026-01-01T00:00:00+00:00",
               "label_reviewed_at": "2026-01-03T00:00:00+00:00", "labels": ["43271"],
               "procedimiento_sistema": "Procedimiento QA", "evidencia_tarifario": evidence(),
               "document_same_admission_verified": True,
               "document_available_at": "2026-01-01T00:00:00+00:00",
               "document_snapshot_captured_at": "2026-01-01T01:00:00+00:00",
               "document_hash_verified_at": "2026-01-01T02:00:00+00:00"}
        self.assertIsNone(validate_record(row, data_kind="synthetic"))
        self.assertEqual(validate_record({**row, "labels": ["70200003"]}, data_kind="synthetic"), "etiquetas_invalidas")


if __name__ == "__main__":
    unittest.main()
