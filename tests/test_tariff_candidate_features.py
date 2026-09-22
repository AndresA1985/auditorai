"""Synthetic feature/transport contracts; never load settings, DB or artifacts."""
from copy import deepcopy
import unittest

from pydantic import ValidationError

from app.features import (
    FEATURE_SCHEMA_VERSION, build_features, build_prediction_text, build_training_text,
)
from app.schemas import PrediccionRequest


def document(**changes):
    evidence = {
        'estado': 'extraido', 'fuente': 'texto', 'documento_sha256': 'a' * 64,
        'codigos': [{'codigo': '43271', 'descripcion': 'Procedimiento documental QA A',
                     'origen': 'texto', 'pagina': 1}],
        'advertencias': [],
    }
    evidence.update(changes)
    return evidence


def payload(**changes):
    value = {
        'id_agenda': 101, 'id_hc': 201, 'id_empresa': 'QA-COMPANY',
        'procedimiento_sistema': 'Procedimiento clínico QA A',
        'hallazgos_conclusion': 'Hallazgos clínicos QA originales.',
        'descripcion_estudio_013': 'Estudio clínico QA original.',
    }
    value.update(changes)
    return value


class TariffCandidateFeatureContractTests(unittest.TestCase):
    def test_version_is_explicit_and_no_document_preserves_three_field_legacy_text(self):
        self.assertEqual(FEATURE_SCHEMA_VERSION, 'tariff_features_v1')
        source = payload()
        legacy = ' '.join(source[key] for key in (
            'procedimiento_sistema', 'hallazgos_conclusion', 'descripcion_estudio_013')).strip()
        self.assertEqual(build_features(source), legacy)
        self.assertEqual(build_prediction_text(PrediccionRequest(**source)), legacy)

    def test_default_features_contain_document_tariff_and_intact_clinical_text(self):
        source = payload(evidencia_tarifario=document())
        text = build_features(source)
        self.assertIn('43271', text)
        self.assertIn('Procedimiento documental QA A', text)
        self.assertIn(source['hallazgos_conclusion'], text)
        self.assertIn(source['descripcion_estudio_013'], text)

    def test_document_only_is_a_real_ablation_without_clinical_or_technical_text(self):
        source = payload(evidencia_tarifario=document(), informe_tecnico_justificacion='Informe técnico clínico QA')
        text = build_features(source, mode='document_only')
        self.assertIn('43271', text)
        for key in ('procedimiento_sistema', 'hallazgos_conclusion', 'descripcion_estudio_013', 'informe_tecnico_justificacion'):
            self.assertNotIn(source[key], text)

    def test_clinical_only_ablation_does_not_use_document_codes_or_description(self):
        source = payload(evidencia_tarifario=document())
        text = build_features(source, mode='clinical')
        self.assertNotIn('43271', text)
        self.assertNotIn('Procedimiento documental QA A', text)
        self.assertIn(source['hallazgos_conclusion'], text)

    def test_training_and_inference_construct_identical_features_in_each_mode(self):
        source = payload(evidencia_tarifario=document(fuente='ocr', codigos=[{'codigo':'43271',
            'descripcion':'Procedimiento documental QA A', 'origen':'ocr', 'pagina':1, 'confianza_ocr':96.15}]),
                         informe_tecnico_justificacion='Informe técnico QA para revisión.')
        request = PrediccionRequest(**source)
        for mode in ('clinical', 'document_only', 'clinical_document'):
            with self.subTest(mode=mode):
                self.assertEqual(build_training_text(source, mode=mode), build_prediction_text(request, mode=mode))
                self.assertEqual(build_features(source, mode=mode), build_prediction_text(request, mode=mode))

    def test_changing_auditor_decisions_and_labels_cannot_change_training_features(self):
        source = payload(evidencia_tarifario=document())
        expected = build_training_text(source)
        for labels in (
            {'codigo_grupo_auditor': '43264', 'honorario_auditor': '100', 'tiempo_anestesia': '8'},
            {'procedimiento_auditor': 'DECISION POSTERIOR QA', 'nombre_procedimiento': 'ETIQUETA QA',
             'observacion_auditor': 'APROBAR CODIGO 43264', 'labels': ['43264']},
        ):
            self.assertEqual(build_training_text({**source, **labels}), expected)

    def test_identifiers_document_hash_and_actor_are_not_predictive_features(self):
        source = payload(evidencia_tarifario=document(), cedula='QA-PERSONAL-ID', paciente='QA-PRIVATE-NAME',
                         actor_user_id='QA-AUDITOR-ID', validation_id='CV2026092543271')
        text = build_features(source)
        for marker in ('QA-PERSONAL-ID', 'QA-PRIVATE-NAME', 'QA-AUDITOR-ID', 'CV2026092543271', 'a' * 64):
            self.assertNotIn(marker, text)

    def test_malformed_document_tokens_never_become_a_tariff_reference(self):
        for token in ('CV2026092543271', '9914327199', '432710000', '43271 43264', '<script>43271</script>'):
            with self.subTest(token=token):
                source = payload(evidencia_tarifario=document(codigos=[{'codigo': token}]))
                text = build_features(source, mode='document_only')
                self.assertNotIn('43271', text)
                with self.assertRaises(ValidationError):
                    PrediccionRequest(**source)

    def test_unlabelled_metadata_numbers_do_not_create_a_tariff_when_codes_are_empty(self):
        source = payload(evidencia_tarifario=document(codigos=[], codigo_validacion='CV2026092543271',
                         cedula='9914327199', costo='43271', fecha='2026-09-18'))
        self.assertNotIn('43271', build_features(source, mode='document_only'))

    def test_absent_document_only_does_not_silently_fall_back_to_clinical_predictors(self):
        self.assertEqual(build_features(payload(), mode='document_only'), '')

    def test_unknown_mode_fails_instead_of_changing_experiment_semantics(self):
        with self.assertRaises(ValueError):
            build_features(payload(evidencia_tarifario=document()), mode='invented_mode')

    def test_builders_do_not_mutate_the_snapshot_or_documentary_source(self):
        source = payload(evidencia_tarifario=document(), informe_tecnico_justificacion='Informe QA opcional')
        before = deepcopy(source)
        build_features(source)
        build_training_text(source)
        build_prediction_text(PrediccionRequest(**source))
        self.assertEqual(source, before)

    def test_gen_transition_prefix_is_replaced_once_without_duplicate_tariff_features(self):
        original = payload(evidencia_tarifario=document())
        transitioned = {**original, 'procedimiento_sistema':
            '[DATOS DEL DOCUMENTO DE VALIDACION: REFERENCIA TARIFARIA PRINCIPAL]\n'
            'Código tarifario del documento: 43271 — Procedimiento documental QA A\n'
            '[FIN DATOS DOCUMENTALES; CONTRASTAR CON LOS HALLAZGOS Y REVISAR DIFERENCIAS]\n'
            + original['procedimiento_sistema']}
        self.assertEqual(build_features(original), build_features(transitioned))
        self.assertEqual(build_features(transitioned).count('43271'), 1)

    def test_gen_transition_technical_report_and_structured_report_are_not_duplicated(self):
        original = payload(evidencia_tarifario=document(), informe_tecnico_justificacion='Informe técnico QA único.')
        transitioned = {**original, 'descripcion_estudio_013': original['descripcion_estudio_013'] +
            '\n[INFORME TECNICO APORTADO PARA REVISION]\n' + original['informe_tecnico_justificacion']}
        self.assertEqual(build_training_text(original), build_prediction_text(PrediccionRequest(**transitioned)))
        self.assertEqual(build_features(transitioned).count(original['informe_tecnico_justificacion']), 1)

    def test_document_description_cannot_introduce_cv_personal_identifier_or_a_second_tariff(self):
        source = payload(evidencia_tarifario=document(codigos=[{'codigo':'43271', 'origen':'texto', 'pagina':1,
            'descripcion':'Procedimiento QA CV2026092543271 9914327199 43264'}]))
        text = build_features(source, mode='document_only')
        self.assertEqual(text.count('43271'), 1)
        self.assertNotIn('CV2026092543271', text)
        self.assertNotIn('9914327199', text)
        self.assertNotIn('43264', text)

    def test_schema_rejects_ocr_source_with_native_origin_and_extracted_state_without_codes(self):
        for doc in (document(fuente='ocr'), document(codigos=[], fuente='ninguna')):
            with self.assertRaises(ValidationError): PrediccionRequest(**payload(evidencia_tarifario=doc))


if __name__ == '__main__':
    unittest.main()
