import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

# La ruta TF-IDF probada no necesita los runtimes pesados opcionales.
if 'torch' not in sys.modules:
    sys.modules['torch'] = types.ModuleType('torch')
if 'transformers' not in sys.modules:
    transformers_stub = types.ModuleType('transformers')
    transformers_stub.AutoModelForSequenceClassification = object
    transformers_stub.AutoTokenizer = object
    sys.modules['transformers'] = transformers_stub

from app.anestesia_model import aplicar_tiempos_anestesia
from app.honorarios_model import aplicar_honorarios
from app.ml_model import anexar_justificaciones_codigo
from app.schemas import CodigoScore, EvidenciaRanking, PrediccionPayload, PrediccionRequest


class VectorizerFalso:
    def transform(self, textos):
        return list(textos)


class ClassifierFalso:
    def __init__(self, classes, full_scores, fragment_scores):
        self.classes_ = np.asarray(classes)
        self.full_scores = np.asarray(full_scores)
        self.fragment_scores = np.asarray(fragment_scores)

    def predict_proba(self, textos):
        if len(textos) == 1:
            return np.asarray([self.full_scores])
        return np.asarray(self.fragment_scores[:len(textos)])


def artifact(classes, full_scores, fragment_scores):
    return {
        'engine': 'tfidf_logreg',
        'vectorizer': VectorizerFalso(),
        'classifier': ClassifierFalso(classes, full_scores, fragment_scores),
    }


class ThreeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.req = PrediccionRequest(
            id_agenda=1,
            procedimiento_sistema='COLONOSCOPIA',
            hallazgos_conclusion=(
                'SE INTRODUCE VIDEOCOLONOSCOPIO HASTA CIEGO. '
                'SE TOMAN BIOPSIAS ESCALONADAS DE COLON.'
            ),
        )
        ranking = [
            {'codigo': '45380', 'score': 0.11, 'selected': True},
            {'codigo': '43239', 'score': 0.09, 'selected': False},
        ]
        self.prediccion = {
            'codigos': ['45380'],
            'codigo_scores': {'45380': 0.11},
            'codigo_ranking': anexar_justificaciones_codigo(
                ranking, self.req.hallazgos_conclusion,
                {'45380': 'SE TOMAN BIOPSIAS ESCALONADAS DE COLON.'},
            ),
            'honorarios_codigo': {},
            'honorario': '',
            'tiempo_anestesia': '',
            'tiempos_anestesia_codigo': {},
            'nombre_procedimiento': 'COLONOSCOPIA',
            'observacion_auditor': 'Anterior',
        }

    def test_tres_evidencias_para_todos_los_codigos_con_scores_independientes(self):
        honorarios = artifact(['50', '100'], [0.2, 0.8], [[0.3, 0.7], [0.1, 0.9]])
        anestesia = artifact(['1', '2'], [0.25, 0.75], [[0.4, 0.6], [0.2, 0.8]])
        with patch('app.honorarios_model.cargar_modelo_honorarios', return_value=honorarios):
            resultado = aplicar_honorarios(self.req, self.prediccion)
        with patch('app.anestesia_model.cargar_modelo_anestesia', return_value=anestesia):
            resultado = aplicar_tiempos_anestesia(self.req, resultado)

        self.assertEqual(len(resultado['codigo_ranking']), 2)
        for item in resultado['codigo_ranking']:
            self.assertEqual(
                set(item['evidencias']), {'codigo', 'honorario', 'tiempo_anestesia'}
            )
            self.assertIn(item['evidencias']['honorario']['texto_soporte'],
                          self.req.hallazgos_conclusion)
            self.assertIn(item['evidencias']['tiempo_anestesia']['texto_soporte'],
                          self.req.hallazgos_conclusion)
            self.assertEqual(item['evidencias']['honorario']['porcentaje_ranking'], 80.0)
            self.assertEqual(
                item['evidencias']['tiempo_anestesia']['porcentaje_ranking'], 75.0
            )
            self.assertNotEqual(
                item['score'], item['evidencias']['honorario']['score_ranking']
            )
        payload = PrediccionPayload(**resultado)
        self.assertEqual(payload.codigo_ranking[0].evidencias.honorario.valor_sugerido, '100')
        self.assertEqual(
            payload.codigo_ranking[0].evidencias.tiempo_anestesia.valor_sugerido, '2'
        )

    def test_observacion_dinamica_solo_para_codigos_seleccionados(self):
        honorarios = artifact(['50', '100'], [0.2, 0.8], [[0.3, 0.7], [0.1, 0.9]])
        anestesia = artifact(['1', '2'], [0.25, 0.75], [[0.4, 0.6], [0.2, 0.8]])
        with patch('app.honorarios_model.cargar_modelo_honorarios', return_value=honorarios):
            resultado = aplicar_honorarios(self.req, self.prediccion)
        with patch('app.anestesia_model.cargar_modelo_anestesia', return_value=anestesia):
            resultado = aplicar_tiempos_anestesia(self.req, resultado)
        observacion = resultado['observacion_auditor']
        self.assertIn('Explicacion integral para el codigo 45380', observacion)
        self.assertNotIn('codigo 43239', observacion)
        self.assertIn('Honorario sugerido: 100 %; puntaje de ranking sobre el texto completo: 80.00%', observacion)
        self.assertIn('Tiempo sugerido: 2 horas; puntaje de ranking sobre el texto completo: 75.00%', observacion)
        self.assertIn('no calibrados', observacion)
        self.assertEqual(
            resultado['codigo_ranking'][0]['evidencias']['tiempo_anestesia']['unidad'],
            'horas',
        )

    def test_schema_acepta_limites_de_scores_y_porcentajes(self):
        base = {
            'disponible': True,
            'texto_soporte': 'Fragmento literal suficiente.',
            'justificacion': 'Requiere validacion.',
            'fuente': 'modelo_prueba',
        }
        minimo = EvidenciaRanking(**base, score_ranking=0, porcentaje_ranking=0)
        maximo = EvidenciaRanking(**base, score_ranking=1, porcentaje_ranking=100)
        self.assertEqual(minimo.score_ranking, 0)
        self.assertEqual(minimo.porcentaje_ranking, 0)
        self.assertEqual(maximo.score_ranking, 1)
        self.assertEqual(maximo.porcentaje_ranking, 100)

    def test_schema_rechaza_scores_y_porcentajes_fuera_de_rango(self):
        base = {
            'disponible': True,
            'justificacion': 'Requiere validacion.',
            'fuente': 'modelo_prueba',
        }
        casos = (
            {'score_ranking': -0.0001, 'porcentaje_ranking': 0},
            {'score_ranking': 1.0001, 'porcentaje_ranking': 100},
            {'score_ranking': 0, 'porcentaje_ranking': -0.0001},
            {'score_ranking': 1, 'porcentaje_ranking': 100.0001},
        )
        for valores in casos:
            with self.subTest(valores=valores):
                with self.assertRaises(ValidationError):
                    EvidenciaRanking(**base, **valores)

    def test_fragmentos_irrelevantes_no_son_evidencia_de_honorario_ni_anestesia(self):
        scores_fragmentos = [[0.99, 0.01], [0.98, 0.02]]
        honorarios = artifact(['50', '100'], [0.2, 0.8], scores_fragmentos)
        anestesia = artifact(['1', '2'], [0.25, 0.75], scores_fragmentos)
        with patch('app.honorarios_model.cargar_modelo_honorarios', return_value=honorarios):
            resultado = aplicar_honorarios(self.req, self.prediccion)
        with patch('app.anestesia_model.cargar_modelo_anestesia', return_value=anestesia):
            resultado = aplicar_tiempos_anestesia(self.req, resultado)
        item = resultado['codigo_ranking'][0]
        for key in ('honorario', 'tiempo_anestesia'):
            evidencia = item['evidencias'][key]
            self.assertFalse(evidencia['disponible'])
            self.assertIsNone(evidencia['texto_soporte'])
            self.assertIn(
                'no se encontro evidencia textual especifica suficiente',
                evidencia['justificacion'],
            )
            self.assertIsNotNone(evidencia['score_ranking'])

    def test_error_de_un_modelo_degrada_solo_su_evidencia(self):
        class ClassifierConError:
            classes_ = np.asarray(['50', '100'])

            def predict_proba(self, textos):
                raise RuntimeError('fallo simulado')

        roto = {
            'engine': 'tfidf_logreg',
            'vectorizer': VectorizerFalso(),
            'classifier': ClassifierConError(),
        }
        with patch('app.honorarios_model.cargar_modelo_honorarios', return_value=roto):
            resultado = aplicar_honorarios(self.req, self.prediccion)
        evidencia = resultado['codigo_ranking'][0]['evidencias']['honorario']
        self.assertFalse(evidencia['disponible'])
        self.assertIsNone(evidencia['score_ranking'])
        self.assertIn('No fue posible calcular', evidencia['justificacion'])
        self.assertTrue(resultado['codigo_ranking'][0]['evidencias']['codigo'])

    def test_schema_legacy_sin_evidencias_conserva_compatibilidad(self):
        dumped = CodigoScore(
            codigo='45380', score=0.2, selected=True
        ).model_dump(exclude_none=True)
        self.assertNotIn('evidencias', dumped)
        self.assertEqual(dumped['score'], 0.2)

    def test_fallback_sin_modelo_declara_no_disponible(self):
        with patch('app.honorarios_model.cargar_modelo_honorarios', return_value=None):
            resultado = aplicar_honorarios(self.req, self.prediccion)
        evidencia = resultado['codigo_ranking'][0]['evidencias']['honorario']
        self.assertFalse(evidencia['disponible'])
        self.assertIsNone(evidencia['score_ranking'])
        self.assertIsNone(evidencia['texto_soporte'])


if __name__ == '__main__':
    unittest.main()
