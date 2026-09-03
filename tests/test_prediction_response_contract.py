import os
import unittest

import numpy as np
from pydantic import ValidationError

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

from app.ml_model import anexar_justificaciones_codigo, fragmentos_clinicos, metricas_modelo_desde_artefacto
from app.schemas import CodigoScore, MetricasModelo, PrediccionPayload
from scripts.train_multilabel import metricas_multilabel


def payload_base(**overrides):
    data = {"codigos": ["45380"], "codigo_ranking": [
        {"codigo": "45380", "score": 0.1351, "selected": True}],
        "honorarios_codigo": {}, "honorario": "", "observacion_auditor": "Validar."}
    data.update(overrides)
    return data


def metricas(f1_macro=0.8421, f1_weighted=0.8573, cantidad_muestras=1250):
    return {"f1_macro": f1_macro, "f1_weighted": f1_weighted,
        "version_modelo": "auditoria-codigos-v1.4.0", "conjunto_evaluacion": "test_holdout",
        "fecha_evaluacion": "2026-09-03", "cantidad_muestras": cantidad_muestras}


class PredictionResponseContractTests(unittest.TestCase):
    def test_respuesta_con_metricas(self):
        payload = PrediccionPayload(**payload_base(metricas_modelo=metricas()))
        self.assertEqual(payload.metricas_modelo.f1_macro, 0.8421)
        self.assertEqual(payload.codigo_ranking[0].score, 0.1351)

    def test_respuesta_sin_metricas_conserva_ranking(self):
        dumped = PrediccionPayload(**payload_base()).model_dump(exclude_none=True)
        self.assertNotIn("metricas_modelo", dumped)
        self.assertEqual(dumped["codigo_ranking"][0]["score"], 0.1351)

    def test_f1_acepta_limites_cero_y_uno(self):
        self.assertEqual(MetricasModelo(**metricas(f1_macro=0, f1_weighted=0)).f1_macro, 0)
        self.assertEqual(MetricasModelo(**metricas(f1_macro=1, f1_weighted=1)).f1_macro, 1)

    def test_fuera_de_rango_y_muestra_cero_son_rechazados(self):
        for valores in (metricas(f1_macro=-0.01), metricas(f1_weighted=1.01), metricas(cantidad_muestras=0)):
            with self.assertRaises(ValidationError):
                MetricasModelo(**valores)

    def test_score_documentado_como_ranking_no_calibrado(self):
        descripcion = CodigoScore.model_json_schema()["properties"]["score"]["description"]
        self.assertIn("ordenar", descripcion)
        self.assertIn("No es F1", descripcion)
        self.assertIn("probabilidad calibrada", descripcion)

    def test_justificacion_preserva_cita_literal_score_selected_y_entrada(self):
        hallazgo = "EL ESOFAGO ESTA NORMAL.\nFRASE  CON  DOS ESPACIOS PARA EL PROCEDIMIENTO."
        soporte = fragmentos_clinicos(hallazgo)[1]
        ranking = [{"codigo": "45380", "score": 0.1351, "selected": True},
                   {"codigo": "43239", "score": 0.12, "selected": False}]
        resultado = anexar_justificaciones_codigo(
            ranking, hallazgo, {"45380": soporte, "43239": soporte})
        for item in resultado:
            self.assertIn(item["texto_soporte"], hallazgo)
            self.assertIn(item["codigo"], item["justificacion"])
        self.assertEqual(resultado[0]["score"], 0.1351)
        self.assertTrue(resultado[0]["selected"])
        self.assertEqual(ranking[0], {"codigo": "45380", "score": 0.1351, "selected": True})

    def test_cita_inventada_y_hallazgo_vacio_degradan_honestamente(self):
        ranking = [{"codigo": "45380", "score": 0.1351, "selected": True}]
        for hallazgo, soportes in (("Hallazgo real suficiente.", {"45380": "Inventado."}), ("", {})):
            item = anexar_justificaciones_codigo(ranking, hallazgo, soportes)[0]
            self.assertIsNone(item["texto_soporte"])
            self.assertIn("no se encontro evidencia textual especifica suficiente", item["justificacion"])

    def test_metricas_salen_solo_del_holdout_completo(self):
        artefacto = {"training_scope": "train_split", "model_version": "auditoria-codigos-v1.4.0",
            "evaluated_at": "2026-09-03T10:30:00", "evaluation": {"evaluated": True,
            "final_test_metrics": {"dataset": "test_holdout", "size": 1250,
            "f1_macro": 0.8421, "f1_weighted": 0.8573}}}
        resultado = metricas_modelo_desde_artefacto(artefacto)
        self.assertEqual(resultado["f1_macro"], 0.8421)
        self.assertEqual(resultado["cantidad_muestras"], 1250)

    def test_legacy_incompleto_y_refit_all_omiten_metricas(self):
        refit = {"training_scope": "all_rows_after_evaluation", "model_version": "v1",
            "evaluated_at": "2026-09-03", "evaluation": {"evaluated": True,
            "final_test_metrics": {"dataset": "test_holdout", "size": 10,
            "f1_macro": 0.5, "f1_weighted": 0.5}}}
        self.assertIsNone(metricas_modelo_desde_artefacto(refit))
        self.assertIsNone(metricas_modelo_desde_artefacto({"version": 1}))

    def test_entrenamiento_calcula_macro_y_weighted(self):
        resultado = metricas_multilabel(
            np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 0]]))
        self.assertEqual(resultado["f1_macro"], 0.5)
        self.assertEqual(resultado["f1_weighted"], 0.5)


if __name__ == "__main__":
    unittest.main()
