import unittest

import numpy as np

from app.ml_model import soportes_por_codigo


class VectorizerFalso:
    def transform(self, fragmentos):
        return fragmentos


class ClassifierFalso:
    def __init__(self, probabilidades):
        self.probabilidades = np.asarray(probabilidades)

    def predict_proba(self, matriz):
        return self.probabilidades


class LabelBinarizerFalso:
    classes_ = np.asarray(["45380"])


class SupportThresholdTests(unittest.TestCase):
    def artifact(self, probabilidades, threshold=0.45):
        return {
            "model": "tfidf_logistic_regression_multilabel",
            "threshold": threshold,
            "vectorizer": VectorizerFalso(),
            "classifier": ClassifierFalso(probabilidades),
            "label_binarizer": LabelBinarizerFalso(),
        }

    def test_texto_irrelevante_bajo_umbral_no_se_presenta_como_evidencia(self):
        ranking = [{"codigo": "45380", "score": 0.2, "selected": False}]
        soportes = soportes_por_codigo(
            self.artifact([[0.01], [0.02]]), ranking,
            "Texto clinico irrelevante. Otro fragmento sin relacion.",
        )
        self.assertEqual(soportes, {})

    def test_selecciona_literal_con_mayor_score_solo_si_supera_umbral(self):
        hallazgo = "Primer fragmento clinico. Segundo  fragmento  relevante."
        ranking = [{"codigo": "45380", "score": 0.8, "selected": True}]
        soportes = soportes_por_codigo(self.artifact([[0.2], [0.8]]), ranking, hallazgo)
        self.assertEqual(soportes["45380"], "Segundo  fragmento  relevante.")
        self.assertIn(soportes["45380"], hallazgo)


if __name__ == "__main__":
    unittest.main()
