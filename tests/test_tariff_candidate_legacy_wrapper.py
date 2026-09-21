"""Exercise real wrapper functions with in-memory synthetic artifacts only.

The AST loader intentionally excludes every module import and model-loader body,
so this test cannot import app.config/main/db or read an existing model.
"""
import ast
from copy import deepcopy
from datetime import date
import math
import operator
from pathlib import Path
import re
from types import SimpleNamespace
from typing import List, Tuple
import unittest

from app.features import FEATURE_SCHEMA_VERSION, build_prediction_text
from app.schemas import PrediccionRequest
from app.tariff_evidence import annotate_documentary_evidence

APP = Path(__file__).resolve().parents[1] / 'app'


def functions_from_source(file, names, globals):
    tree = ast.parse((APP / file).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError('Candidate wrapper public contract is missing.')
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(APP / file), 'exec'), globals)
    return globals


def req(document=False):
    source = {'id_agenda':101, 'procedimiento_sistema':'Procedimiento clínico QA',
        'hallazgos_conclusion':'Hallazgo QA original. Frase clínica  con dos espacios.',
        'descripcion_estudio_013':'Estudio QA original.'}
    if document:
        source['evidencia_tarifario'] = {'estado':'extraido', 'fuente':'texto', 'documento_sha256':'c'*64,
            'codigos':[{'codigo':'43271', 'descripcion':'Procedimiento documental QA', 'origen':'texto', 'pagina':1}],
            'advertencias':[]}
    return PrediccionRequest(**source)


def predicted(codes=None):
    return {'codigos':['43271'] if codes is None else codes, 'codigo_scores':{'43271':.21},
        'codigo_ranking':[{'codigo':'43271','score':.21,'selected':True},
                          {'codigo':'43264','score':.19,'selected':False}],
        'honorarios_codigo':{'43271':'100'}, 'honorario':'43271(100)',
        'tiempos_anestesia_codigo':{'43271':'2'}, 'tiempo_anestesia':'2',
        'nombre_procedimiento':'Procedimiento QA', 'observacion_auditor':'Validar manualmente.'}


def wrapper(artifact, proposal=None):
    base_functions = functions_from_source('ml_model_base.py',
        {'es_artefacto_tarifario','texto_request','min_labels_del_artefacto','prediccion_candidata_vacia'},
        {'FEATURE_SCHEMA_VERSION':FEATURE_SCHEMA_VERSION, 'build_prediction_text':build_prediction_text,
         'PrediccionRequest':PrediccionRequest})
    calls = []
    original = predicted() if proposal is None else proposal
    def choose_model(synthetic_artifact, text):
        calls.append(text)
        return deepcopy(original), .21
    base = SimpleNamespace(**{name:base_functions[name] for name in (
        'es_artefacto_tarifario','texto_request','min_labels_del_artefacto','prediccion_candidata_vacia')})
    base.MODEL_PATH = Path('/synthetic-no-artifact-is-read')
    base.cargar_modelo = lambda path: artifact
    base.predecir_multilabel = choose_model
    base.predecir_transformer_multilabel = choose_model
    base.predecir_vecino = choose_model
    namespace = functions_from_source('ml_model.py',
        {'fragmentos_clinicos','evidencia_no_disponible','anexar_justificaciones_codigo',
         'metricas_modelo_desde_artefacto','predecir_con_modelo'},
        {'_base':base, 'Path':Path, 'PrediccionRequest':PrediccionRequest,
         'List':List, 'Tuple':Tuple, 're':re, 'math':math, 'operator':operator, 'date':date,
         'MAX_SUPPORT_LENGTH':500, 'SCORE_DECIMALS':6,
         'annotate_documentary_evidence':annotate_documentary_evidence,
         'soportes_por_codigo':lambda artifact, ranking, hallazgo: {'43271':'Frase clínica  con dos espacios.'}})
    return namespace, base, calls


def evaluated_artifact(**changes):
    artifact = {'model':'tfidf_logistic_regression_multilabel', 'training_scope':'train_split',
        'model_version':'synthetic-QA-v1', 'evaluated_at':'2026-09-18T12:00:00+00:00',
        'clinical_performance_validated':True,
        'clinical_validation_gate':'approved_independent_operator_review',
        'evaluation':{'evaluated':True, 'final_test_metrics':{'dataset':'test_holdout','size':3,
            'f1_macro':.4,'f1_weighted':.4}}}
    artifact.update(changes)
    return artifact


class TariffLegacyWrapperContractTests(unittest.TestCase):
    def test_old_artifact_prediction_and_scores_are_preserved_when_structured_evidence_is_added(self):
        namespace, base, calls = wrapper({'model':'tfidf_logistic_regression_multilabel'})
        legacy, legacy_score = namespace['predecir_con_modelo'](req(False))
        documentary, documentary_score = namespace['predecir_con_modelo'](req(True))
        self.assertEqual(calls[0], calls[1])
        self.assertNotIn('43271', calls[1])
        self.assertEqual(legacy_score, documentary_score)
        for key, value in legacy.items():
            self.assertEqual(documentary[key], value)
        self.assertEqual(documentary['comparacion_ranking_tarifario'][0]['score_ranking'], .21)
        self.assertEqual(documentary['comparacion_ranking_tarifario'][1]['score_ranking'], .19)
        self.assertEqual(documentary['discrepancias'], [])
        self.assertNotIn('feature_schema_version', documentary)

    def test_old_artifact_without_pdf_has_no_new_documentary_fields(self):
        namespace, base, calls = wrapper({'model':'tfidf_logistic_regression_multilabel'})
        value, score = namespace['predecir_con_modelo'](req())
        for key in ('evidencia_tarifario','comparacion_ranking_tarifario','requiere_revision','abstencion'):
            self.assertNotIn(key, value)
        self.assertEqual(value['codigos'], ['43271'])
        self.assertEqual(value['codigo_scores'], {'43271':.21})
        self.assertEqual([item['score'] for item in value['codigo_ranking']], [.21,.19])

    def test_document_disagreement_annotates_old_artifact_without_replacing_its_choice(self):
        namespace, base, calls = wrapper({'model':'tfidf_logistic_regression_multilabel'}, predicted(['43264']))
        value, score = namespace['predecir_con_modelo'](req(True))
        self.assertEqual(value['codigos'], ['43264'])
        self.assertEqual(value['codigo_scores'], {'43271':.21})
        self.assertTrue(value['requiere_revision'])
        self.assertTrue(value['discrepancias'])
        self.assertEqual(score, .21)

    def test_versioned_candidate_uses_the_shared_features_and_keeps_the_score_semantics(self):
        namespace, base, calls = wrapper({'model':'tfidf_logistic_regression_multilabel',
            'feature_schema_version':FEATURE_SCHEMA_VERSION, 'feature_mode':'clinical_document'})
        value, score = namespace['predecir_con_modelo'](req(True))
        self.assertEqual(calls, [build_prediction_text(req(True))])
        self.assertIn('43271', calls[0])
        self.assertEqual(value['feature_schema_version'], FEATURE_SCHEMA_VERSION)
        self.assertEqual(score, .21)
        self.assertEqual(value['codigo_ranking'][0]['score'], .21)

    def test_unrecognized_artifact_schema_cannot_fall_back_to_an_incompatible_legacy_representation(self):
        namespace, base, calls = wrapper({'model':'tfidf_logistic_regression_multilabel',
            'feature_schema_version':'future-incompatible-schema'})
        with self.assertRaises(ValueError): namespace['predecir_con_modelo'](req(True))
        self.assertEqual(calls, [])

    def test_document_only_candidate_with_no_document_abstains_without_calling_a_classifier(self):
        namespace, base, calls = wrapper({'model':'tfidf_logistic_regression_multilabel',
            'feature_schema_version':FEATURE_SCHEMA_VERSION, 'feature_mode':'document_only', 'allow_abstention':True})
        value, score = namespace['predecir_con_modelo'](req())
        self.assertEqual(calls, [])
        self.assertEqual(value['codigos'], [])
        self.assertTrue(value['abstencion'])
        self.assertTrue(value['requiere_revision'])
        self.assertEqual(value['motivo_abstencion'], 'sin_features_utilizables')
        self.assertEqual(score, 0)

    def test_legacy_transformer_dispatch_and_documentary_option_preserve_codes_scores_and_financial_outputs(self):
        original = predicted(['43264'])
        original['codigo_ranking'][0]['selected'] = False
        original['codigo_ranking'][1]['selected'] = True
        original.update({'honorarios_codigo':{'43264':'75'}, 'honorario':'43264(75)',
            'tiempos_anestesia_codigo':{'43264':'4'}, 'tiempo_anestesia':'4'})
        namespace, base, calls = wrapper({'model':'transformer_multilabel'}, original)
        transformer_calls = []
        def transformer(artifact, text):
            transformer_calls.append(text)
            return deepcopy(original), .37
        def wrong_family(*args):
            self.fail('Transformer artifacts must not dispatch to TF-IDF or embedding predictors.')
        base.predecir_transformer_multilabel = transformer
        base.predecir_multilabel = base.predecir_vecino = wrong_family
        legacy, legacy_score = namespace['predecir_con_modelo'](req(False))
        documentary, documentary_score = namespace['predecir_con_modelo'](req(True))
        self.assertEqual(len(transformer_calls), 2)
        self.assertEqual(transformer_calls[0], transformer_calls[1])
        self.assertNotIn('43271', transformer_calls[1])
        self.assertEqual((legacy_score, documentary_score), (.37, .37))
        for key, value in legacy.items():
            self.assertEqual(documentary[key], value)
        for key in ('codigos', 'codigo_scores', 'honorarios_codigo', 'honorario',
                    'tiempos_anestesia_codigo', 'tiempo_anestesia'):
            self.assertEqual(documentary[key], original[key])
        self.assertEqual([item['score'] for item in documentary['codigo_ranking']], [.21, .19])
        self.assertEqual([item['selected'] for item in documentary['codigo_ranking']], [False, True])
        option = documentary['opciones_tarifario_documental'][0]
        self.assertEqual(option['codigo'], '43271')
        self.assertEqual(option['prioridad'], 'alta')
        self.assertEqual(option['modelo_score'], .21)
        self.assertFalse(option['modelo_selected'])
        self.assertEqual(original['codigos'], ['43264'])

    def test_support_dispatch_uses_the_matching_legacy_model_family_without_loading_any_weights(self):
        calls = []
        def evidence(family):
            def fake(artifact, fragments, codes):
                calls.append((family, fragments, codes))
                return {'43271':'Hallazgo QA original.'}
            return fake
        namespace = functions_from_source('ml_model.py', {'fragmentos_clinicos','soportes_por_codigo'},
            {'List':List, 're':re, 'MAX_SUPPORT_LENGTH':500,
             '_soportes_tfidf':evidence('tfidf'), '_soportes_transformer':evidence('transformer')})
        for model, family in (('tfidf_logistic_regression_multilabel','tfidf'), ('transformer_multilabel','transformer')):
            result = namespace['soportes_por_codigo']({'model':model}, [{'codigo':'43271'}], 'Hallazgo QA original.')
            self.assertEqual(result, {'43271':'Hallazgo QA original.'})
            self.assertEqual(calls[-1], (family, ['Hallazgo QA original.'], ['43271']))
        before = len(calls)
        self.assertEqual(namespace['soportes_por_codigo']({'model':'embedding'}, [{'codigo':'43271'}], 'Hallazgo QA original.'), {})
        self.assertEqual(len(calls), before)

    def test_candidate_tfidf_settings_match_the_established_pipeline_and_fit_only_training_rows(self):
        scripts = APP.parent / 'scripts'
        candidate = ast.parse((scripts / 'train_tariff_candidate.py').read_text())
        established = ast.parse((scripts / 'train_multilabel_base.py').read_text())
        vectorizers = [node for node in ast.walk(candidate) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == 'TfidfVectorizer']
        self.assertEqual(len(vectorizers), 1)
        parameters = {item.arg:ast.literal_eval(item.value) for item in vectorizers[0].keywords}
        self.assertEqual(parameters, {'analyzer':'char_wb', 'ngram_range':(3,5), 'lowercase':True,
            'strip_accents':'unicode', 'min_df':2, 'max_features':250000})
        defaults = {}
        for node in ast.walk(established):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
                if node.args and isinstance(node.args[0], ast.Constant):
                    for item in node.keywords:
                        if item.arg == 'default' and isinstance(item.value, ast.Constant):
                            defaults[node.args[0].value] = item.value.value
        self.assertEqual(parameters['ngram_range'], (defaults['--ngram-min'], defaults['--ngram-max']))
        self.assertEqual(parameters['min_df'], defaults['--min-df'])
        self.assertEqual(parameters['max_features'], defaults['--max-features'])
        fits = [node for node in ast.walk(candidate) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == 'fit_transform'
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'vectorizer']
        self.assertEqual(len(fits), 1)
        self.assertIsInstance(fits[0].args[0], ast.ListComp)
        self.assertEqual(ast.unparse(fits[0].args[0].generators[0].iter), 'effective_train')
        transform_sources = [ast.unparse(node.args[0]) for node in ast.walk(candidate)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'transform'
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'vectorizer']
        self.assertIn('[texts[index] for index in validation_indices]', transform_sources)
        self.assertIn('test_texts', transform_sources)

    def test_abstention_is_explicitly_enabled_for_candidates_and_legacy_minimum_labels_is_preserved(self):
        namespace, base, calls = wrapper({'model':'tfidf_logistic_regression_multilabel'})
        self.assertEqual(base.min_labels_del_artefacto({'min_labels':1}), 1)
        self.assertEqual(base.min_labels_del_artefacto({'min_labels':1, 'feature_schema_version':FEATURE_SCHEMA_VERSION,
            'allow_abstention':True}), 0)

    def test_active_artifact_global_metrics_are_only_published_for_the_evaluated_train_split_holdout(self):
        namespace, base, calls = wrapper({'model':'tfidf_logistic_regression_multilabel'})
        read = namespace['metricas_modelo_desde_artefacto']
        result = read(evaluated_artifact())
        self.assertEqual(result['f1_macro'], .4)
        self.assertEqual(result['cantidad_muestras'], 3)
        self.assertEqual(result['conjunto_evaluacion'], 'test_holdout')
        self.assertIsNone(read(evaluated_artifact(training_scope='all_rows_after_evaluation')))
        self.assertIsNone(read({'version':1}))
        for dataset in ('train','validation_holdout'):
            artifact = evaluated_artifact()
            artifact['evaluation']['final_test_metrics']['dataset'] = dataset
            self.assertIsNone(read(artifact))

    def test_active_metric_nan_invalid_ranges_counts_or_missing_provenance_are_omitted(self):
        namespace, base, calls = wrapper({'model':'tfidf_logistic_regression_multilabel'})
        read = namespace['metricas_modelo_desde_artefacto']
        for key, invalid in (('f1_macro',float('nan')),('f1_weighted',float('inf')),
                             ('f1_macro',True),('f1_weighted',False),
                             ('f1_macro',-0.01),('f1_weighted',1.01),('size',0),('size',True)):
            artifact = evaluated_artifact()
            artifact['evaluation']['final_test_metrics'][key] = invalid
            self.assertIsNone(read(artifact))
        for changes in ({'model_version':''}, {'evaluated_at':'invalid-date'}, {'evaluation':{'evaluated':False}}):
            self.assertIsNone(read(evaluated_artifact(**changes)))

    def test_synthetic_holdout_or_explicitly_unvalidated_clinical_metrics_are_never_exposed_as_model_f1(self):
        namespace, base, calls = wrapper({'model':'tfidf_logistic_regression_multilabel'})
        read = namespace['metricas_modelo_desde_artefacto']
        self.assertIsNone(read(evaluated_artifact(dataset_kind='synthetic')))
        self.assertIsNone(read(evaluated_artifact(clinical_performance_validated=False)))
        self.assertIsNone(read(evaluated_artifact(clinical_performance_validated=None)))
        missing = evaluated_artifact()
        missing.pop('clinical_performance_validated')
        self.assertIsNone(read(missing))
        self.assertIsNone(read(evaluated_artifact(clinical_validation_gate='pending_independent_operator_approval')))

    def test_truthy_strings_and_integers_do_not_certify_that_an_artifact_was_evaluated(self):
        namespace, base, calls = wrapper({'model':'tfidf_logistic_regression_multilabel'})
        read = namespace['metricas_modelo_desde_artefacto']
        for invalid in ('true', 'yes', 1, None):
            artifact = evaluated_artifact()
            artifact['evaluation']['evaluated'] = invalid
            self.assertIsNone(read(artifact))


if __name__ == '__main__':
    unittest.main()
