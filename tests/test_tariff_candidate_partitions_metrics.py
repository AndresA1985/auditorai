"""Independent synthetic fixtures for chronology, leakage and honest metrics."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app import tariff_evaluation
from app.tariff_evaluation import (
    bootstrap_intervals, document_cohort_metrics, multilabel_metrics,
    partition_metric_metadata,
)
from scripts.tariff_snapshot_dataset import (
    SnapshotError, clinical_input_hash, freeze_temporal_split, load_snapshot_dataset,
    resolve_split_manifest, shared_pipeline_metadata, timestamp,
)


def record(index):
    predicted = datetime(2026, 9, 1, tzinfo=timezone.utc) + timedelta(days=index)
    def iso(offset):
        return (predicted + timedelta(hours=offset)).isoformat()
    return {'sample_id': f'sample-{index}', 'patient_group': f'patient-{index}',
        'admission_group': f'admission-{index}', 'snapshot_immutable': True,
        'prediction_at': iso(0), 'clinical_snapshot_captured_at': iso(-1), 'clinical_available_at': iso(-2),
        'label_reviewed_at': iso(24), 'labels': ['43271'], 'label_origin': 'auditor_reviewed',
        'source_revision': 'prospective_capture', 'provenance_review_id': 'review-QA',
        'procedimiento_sistema': 'Procedimiento QA A', 'hallazgos_conclusion': 'Hallazgos QA originales.',
        'descripcion_estudio_013': f'Descripción de estudio QA original {index}.',
        'evidencia_tarifario': {'estado': 'extraido', 'fuente': 'texto',
            'documento_sha256': hashlib.sha256(f'doc-QA-{index}'.encode()).hexdigest(),
            'codigos': [{'codigo': '43271', 'descripcion': 'Procedimiento documental QA A', 'origen': 'texto', 'pagina': 1}]},
        'document_same_admission_verified': True, 'document_available_at': iso(-2),
        'document_snapshot_captured_at': iso(-1), 'document_hash_verified_at': iso(-.5)}


def load(rows, *, kind='synthetic', **top_changes):
    source = {'schema_version': 'tariff_snapshot_v1', 'deidentified': True, 'data_kind': kind, 'records': rows}
    source.update(top_changes)
    with tempfile.TemporaryDirectory(prefix='auditorai-synthetic-contract-') as directory:
        file = Path(directory) / 'synthetic.json'
        raw = json.dumps(source, sort_keys=True).encode()
        file.write_bytes(raw)
        result = load_snapshot_dataset(file)
        assert result['manifest_sha256'] == hashlib.sha256(raw).hexdigest()
        assert file.read_bytes() == raw
        return result


class TariffSnapshotContractTests(unittest.TestCase):
    def test_valid_synthetic_snapshot_has_content_hash_and_attested_provenance_limit(self):
        result = load([record(0)])
        self.assertEqual(len(result['records']), 1)
        self.assertEqual(result['excluded_counts'], {})
        self.assertEqual(result['data_kind'], 'synthetic')
        self.assertIn('operator_attestation', result['provenance_scope'])

    def test_top_level_claims_are_required_and_synthetic_is_not_mislabeled_clinical(self):
        for invalid in ({'schema_version': 'old'}, {'deidentified': False}, {'deidentified': 'true'}, {'data_kind': 'production'}):
            with self.subTest(invalid=invalid), self.assertRaises(SnapshotError):
                load([record(0)], **invalid)

    def test_post_prediction_clinical_technical_or_documentary_content_is_excluded(self):
        for field in ('clinical_snapshot_captured_at', 'clinical_available_at', 'document_available_at',
                      'document_snapshot_captured_at', 'document_hash_verified_at', 'technical_available_at'):
            row = record(0)
            row[field] = (timestamp(row['prediction_at']) + timedelta(seconds=1)).isoformat()
            if field == 'technical_available_at':
                row['informe_tecnico_justificacion'] = 'Informe técnico QA'
            with self.subTest(field=field):
                result = load([row])
                self.assertEqual(result['records'], [])
                self.assertEqual(sum(result['excluded_counts'].values()), 1)

    def test_label_before_prediction_naive_timestamps_and_missing_capture_are_excluded(self):
        cases = [('label_reviewed_at', '2026-08-31T23:00:00+00:00'),
                 ('clinical_available_at', '2026-08-31T23:00:00'),
                 ('clinical_snapshot_captured_at', None), ('document_hash_verified_at', None)]
        for key, value in cases:
            with self.subTest(key=key):
                row = record(0); row[key] = value
                self.assertEqual(load([row])['records'], [])

    def test_invalid_or_mismatched_document_identity_cannot_train_as_a_verified_reference(self):
        rows = []
        for index, invalid in enumerate(('different_admission', 'bad_hash', 'cv_token')):
            row = record(index)
            if invalid == 'different_admission': row['document_same_admission_verified'] = False
            if invalid == 'bad_hash': row['evidencia_tarifario']['documento_sha256'] = 'not-a-hash'
            if invalid == 'cv_token': row['evidencia_tarifario']['codigos'][0]['codigo'] = 'CV2026092543271'
            rows.append(row)
        result = load(rows)
        self.assertEqual(result['records'], [])
        self.assertEqual(sum(result['excluded_counts'].values()), 3)

    def test_loader_aligns_document_state_source_codes_and_hash_with_runtime_schema(self):
        invalid_rows = []
        missing_hash = record(0)
        missing_hash['evidencia_tarifario']['documento_sha256'] = None
        invalid_rows.append(missing_hash)
        wrong_source = record(1)
        wrong_source['evidencia_tarifario']['fuente'] = 'ocr'
        invalid_rows.append(wrong_source)
        extracted_without_codes = record(2)
        extracted_without_codes['evidencia_tarifario']['codigos'] = []
        extracted_without_codes['evidencia_tarifario']['fuente'] = 'ninguna'
        invalid_rows.append(extracted_without_codes)
        absent_with_hash = record(3)
        absent_with_hash['evidencia_tarifario'].update(
            {'estado': 'sin_pdf', 'fuente': 'ninguna', 'codigos': []}
        )
        invalid_rows.append(absent_with_hash)
        result = load(invalid_rows)
        self.assertEqual(result['records'], [])
        self.assertEqual(sum(result['excluded_counts'].values()), 4)

        empty_pdf = record(4)
        empty_pdf['evidencia_tarifario'].update(
            {'estado': 'sin_codigos', 'fuente': 'ninguna', 'codigos': []}
        )
        self.assertEqual(len(load([empty_pdf])['records']), 1)

    def test_raw_identity_cv_identifier_or_auditor_decisions_are_excluded_from_snapshot_inputs(self):
        changes = [{'cedula': '9914327199'}, {'procedimiento_auditor': 'DECISION QA'},
                   {'observacion_auditor': 'ETIQUETA QA'}, {'codigo_grupo_auditor': '43264'},
                   {'hallazgos_conclusion': 'QA 9914327199'},
                   {'descripcion_estudio_013': 'QA CV2026092543271'},
                   {'evidencia_tarifario': {**record(0)['evidencia_tarifario'], 'codigos': [
                       {'codigo': '43271', 'descripcion': 'QA qa@example.invalid'}]}}]
        for invalid in changes:
            with self.subTest(fields=tuple(invalid)):
                self.assertEqual(load([{**record(0), **invalid}])['records'], [])

    def test_duplicate_snapshot_id_is_excluded_and_an_independent_snapshot_is_retained(self):
        result = load([record(0), record(0), record(1)])
        self.assertEqual(len(result['records']), 2)
        self.assertEqual(result['excluded_counts']['sample_duplicado'], 1)

    def test_nested_identity_decisions_and_unknown_fields_cannot_hide_behind_deidentified_claim(self):
        for changes in ({'raw_payload': {'cedula':'9914327199'}}, {'metadata': {'codigo_grupo_auditor':'43264'}},
                        {'unexpected_field':'QA'}, {'evidencia_tarifario': {**record(0)['evidencia_tarifario'],
                         'raw_pdf_text':'QA CV2026092543271'}}):
            with self.subTest(fields=tuple(changes)):
                self.assertEqual(load([{**record(0), **changes}])['records'], [])

    def test_available_captured_verified_timestamps_cannot_be_attested_in_reverse_order(self):
        for changes in ({'clinical_available_at': '2026-08-31T23:30:00+00:00'},
                        {'document_available_at': '2026-08-31T23:30:00+00:00'},
                        {'document_hash_verified_at': '2026-08-31T22:30:00+00:00'}):
            self.assertEqual(load([{**record(0), **changes}])['records'], [])

    def test_clinical_dataset_requires_reviewed_provenance_and_auditor_labels(self):
        self.assertEqual(len(load([record(0)], kind='clinical')['records']), 1)
        for changes in ({'source_revision': 'reconstructed_today'}, {'provenance_review_id': None},
                        {'label_origin': 'model_generated'}, {'snapshot_immutable': False}, {'labels': []},
                        {'labels': ['9914327199']}):
            self.assertEqual(load([{**record(0), **changes}], kind='clinical')['records'], [])

    def test_transitive_hash_and_patient_group_stays_in_one_component(self):
        rows = [record(i) for i in range(20)]
        rows[1]['evidencia_tarifario']['documento_sha256'] = rows[0]['evidencia_tarifario']['documento_sha256'].upper()
        rows[2]['patient_group'] = rows[1]['patient_group']
        split = freeze_temporal_split(rows)
        self.assertTrue(any(set([0, 1, 2]) <= set(component) for component in split['components']))
        self.assertTrue(set([0, 1, 2]) <= set(split['train']))
        self.assertEqual(split['excluded'], [])
        self.assertPartitionDisjoint(rows, split)

    def test_transitive_component_crossing_cuts_is_purged_without_leaking_into_test(self):
        rows = [record(i) for i in range(20)]
        rows[14]['evidencia_tarifario']['documento_sha256'] = rows[13]['evidencia_tarifario']['documento_sha256']
        rows[17]['patient_group'] = rows[14]['patient_group']
        split = freeze_temporal_split(rows)
        self.assertEqual(split['excluded'], [13, 14, 17])
        self.assertEqual(split['excluded_counts']['grupo_cruza_corte_temporal'], 3)
        self.assertPartitionDisjoint(rows, split)

    def test_shared_admission_alone_is_enough_to_purge_correlated_observations(self):
        rows = [record(i) for i in range(20)]
        rows[19]['admission_group'] = rows[3]['admission_group']
        split = freeze_temporal_split(rows)
        self.assertEqual(split['excluded'], [3, 19])
        self.assertPartitionDisjoint(rows, split)

    def test_exact_normalized_clinical_duplicates_crossing_cuts_are_purged(self):
        rows = [record(i) for i in range(20)]
        for field in ('procedimiento_sistema', 'hallazgos_conclusion', 'descripcion_estudio_013'):
            rows[19][field] = '  ' + rows[3][field].upper() + '  '
        self.assertEqual(clinical_input_hash(rows[3]), clinical_input_hash(rows[19]))
        split = freeze_temporal_split(rows)
        self.assertEqual(split['excluded'], [3, 19])
        self.assertPartitionDisjoint(rows, split)

    def test_temporal_partitions_are_strictly_ordered_frozen_and_do_not_mutate_records(self):
        rows = [record(i) for i in range(20)]
        before = deepcopy(rows)
        first = freeze_temporal_split(rows)
        self.assertEqual(first, freeze_temporal_split(rows))
        self.assertEqual(rows, before)
        train = [timestamp(rows[i]['prediction_at']) for i in first['train']]
        validation = [timestamp(rows[i]['prediction_at']) for i in first['validation']]
        test = [timestamp(rows[i]['prediction_at']) for i in first['test']]
        self.assertLess(max(train), min(validation))
        self.assertLess(max(validation), min(test))
        self.assertEqual(sorted(first['train'] + first['validation'] + first['test']), list(range(20)))

    def test_common_manifest_records_dataset_split_sources_arguments_and_environment(self):
        rows = [record(index) for index in range(20)]
        root = Path(__file__).resolve().parents[1]
        pipeline = shared_pipeline_metadata(root)
        dataset_sha = 'f' * 64
        manifest = resolve_split_manifest(rows, dataset_sha, pipeline)
        self.assertEqual(manifest['manifest_version'], 'tariff_split_manifest_v1')
        self.assertEqual(manifest['dataset_sha256'], dataset_sha)
        self.assertEqual(manifest['pipeline']['split_arguments'], {
            'validation_size': .15, 'test_size': .15,
        })
        self.assertEqual(set(manifest['pipeline']['source_sha256']), {
            'feature_builder', 'split_builder', 'tfidf_trainer', 'transformer_trainer',
        })
        self.assertIn('python', manifest['pipeline']['environment'])
        with tempfile.TemporaryDirectory(prefix='auditorai-manifest-') as directory:
            path = Path(directory) / 'split_manifest.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            self.assertEqual(resolve_split_manifest(rows, dataset_sha, pipeline, path), manifest)

    def test_group_purge_cannot_silently_produce_an_empty_validation_or_test_partition(self):
        rows = [record(i) for i in range(20)]
        for row in rows: row['patient_group'] = 'patient-correlated'
        with self.assertRaises(SnapshotError): freeze_temporal_split(rows)
        with self.assertRaises(SnapshotError): freeze_temporal_split(rows[:6])
        for validation, test in ((0, .15), (.9, .2), (.15, 0)):
            with self.assertRaises(SnapshotError): freeze_temporal_split(rows, validation, test)

    def assertPartitionDisjoint(self, rows, split):
        keysets = {}
        for name in ('train', 'validation', 'test'):
            selected = [rows[index] for index in split[name]]
            keysets[name] = {
                'patient': {r['patient_group'] for r in selected},
                'admission': {r['admission_group'] for r in selected},
                'hash': {r['evidencia_tarifario']['documento_sha256'].lower() for r in selected},
                'clinical_text': {clinical_input_hash(r) for r in selected},
            }
        for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
            for key in ('patient', 'admission', 'hash', 'clinical_text'):
                self.assertFalse(keysets[left][key] & keysets[right][key], (left, right, key))


class TariffHonestMetricContractTests(unittest.TestCase):
    def test_empty_document_with_ocr_timeout_belongs_to_both_nonexclusive_cohorts(self):
        records = [{
            'labels': ['43271'],
            'evidencia_tarifario': {
                'estado': 'sin_codigos', 'fuente': 'ninguna', 'codigos': [],
                'advertencias': ['tiempo_ocr_agotado'],
            },
        }]
        cohorts = document_cohort_metrics(
            records, [['43271']], [[]], [0]
        )
        self.assertEqual(cohorts['document_empty']['size'], 1)
        self.assertEqual(cohorts['ocr_failed_or_unreadable']['size'], 1)
        self.assertEqual(cohorts['document_present']['size'], 0)
        self.assertEqual(cohorts['document_missing']['size'], 0)

    def test_manual_tp_fp_fn_fixture_matches_reported_micro_macro_and_weighted_scores(self):
        result = multilabel_metrics([['43271'], ['43271', '43264'], ['43264']],
                                    [['43271'], ['43271'], ['43271']])
        self.assertEqual(result['per_code']['43271']['tp'], 2)
        self.assertEqual(result['per_code']['43271']['fp'], 1)
        self.assertEqual(result['per_code']['43264']['fn'], 2)
        self.assertAlmostEqual(result['precision_micro'], 2 / 3)
        self.assertAlmostEqual(result['recall_micro'], 1 / 2)
        self.assertAlmostEqual(result['f1_micro'], 4 / 7)
        self.assertAlmostEqual(result['f1_macro'], .4)
        self.assertAlmostEqual(result['f1_weighted'], .4)
        self.assertAlmostEqual(result['avg_dice'], 5 / 9)
        self.assertAlmostEqual(result['avg_code_overlap'], .5)
        self.assertAlmostEqual(result['exact_code_group_accuracy'], 1 / 3)
        self.assertAlmostEqual(result['unsupported_prediction_rate'], 1 / 3)

    def test_abstention_is_not_reported_as_a_correct_group_or_perfect_precision(self):
        result = multilabel_metrics([['43271'], ['43264']], [[], []])
        self.assertEqual(result['abstention_rate'], 1)
        self.assertEqual(result['f1_micro'], 0)
        self.assertEqual(result['exact_code_group_accuracy'], 0)
        self.assertEqual(result['precision_micro'], 0)

    def test_unknown_test_label_and_unsupported_prediction_cannot_disappear_from_metrics(self):
        result = multilabel_metrics([['43264']], [['99999']], catalog=['43271'])
        self.assertEqual(result['per_code']['43264']['fn'], 1)
        self.assertEqual(result['per_code']['99999']['fp'], 1)
        self.assertEqual(result['unsupported_prediction_rate'], 1)
        self.assertEqual(result['f1_micro'], 0)

    def test_duplicate_codes_do_not_inflate_true_positives(self):
        self.assertEqual(multilabel_metrics([['43271']], [['43271']]),
                         multilabel_metrics([['43271', '43271']], [['43271', '43271']]))

    def test_mismatched_or_empty_evaluation_partition_is_rejected(self):
        for actual, predicted in (([], []), ([['43271']], []), ([['43271']], [[], []])):
            with self.assertRaises(ValueError): multilabel_metrics(actual, predicted)

    def test_public_metric_metadata_requires_positive_holdout_and_keeps_refit_separate(self):
        metrics = {'f1_macro': .4, 'f1_weighted': .4}
        expected = {**metrics, 'size': 3, 'dataset': 'test_holdout'}
        self.assertEqual(partition_metric_metadata(metrics, 3, 'test_holdout'), expected)
        self.assertIsNone(partition_metric_metadata(metrics, 3, 'test_holdout', training_scope='all_rows_after_evaluation'))
        for dataset in ('train', 'validation_holdout', 'synthetic_training'):
            with self.assertRaises(ValueError): partition_metric_metadata(metrics, 3, dataset)
        for size in (0, -1, True, '3'):
            with self.assertRaises(ValueError): partition_metric_metadata(metrics, size, 'test_holdout')

    def test_nan_infinity_out_of_range_and_boolean_f1_are_rejected(self):
        for value in (float('nan'), float('inf'), -0.01, 1.01, True, '0.9', None):
            for key in ('f1_macro', 'f1_weighted'):
                with self.subTest(key=key, value=repr(value)), self.assertRaises(ValueError):
                    partition_metric_metadata({'f1_macro': .4, 'f1_weighted': .4, key: value}, 3, 'test_holdout')

    def test_component_bootstrap_always_resamples_related_visits_together(self):
        original = tariff_evaluation.multilabel_metrics
        calls = []
        def capture(actual, predicted, catalog=None):
            counts = Counter(tuple(row) for row in actual)
            self.assertEqual(counts[('43111',)], counts[('43112',)])
            calls.append(actual)
            return original(actual, predicted, catalog)
        with patch.object(tariff_evaluation, 'multilabel_metrics', side_effect=capture):
            result = bootstrap_intervals([['43111'], ['43112'], ['43271']],
                [['43111'], [], ['43271']], iterations=30, seed=7, independent_groups=[[0, 1], [2]])
        self.assertEqual(len(calls), 30)
        self.assertTrue(all(interval['independent_groups'] == 2 for interval in result.values()))
        self.assertTrue(all(0 <= interval['lower'] <= interval['upper'] <= 1 for interval in result.values()))

    def test_bootstrap_is_repeatable_and_rejects_duplicate_or_missing_group_indices(self):
        actual, predicted = [['43271'], ['43264']], [['43271'], []]
        self.assertEqual(bootstrap_intervals(actual, predicted, iterations=20, seed=7),
                         bootstrap_intervals(actual, predicted, iterations=20, seed=7))
        for groups in ([[0], [0, 1]], [[0]], [[0], [2]]):
            with self.assertRaises(ValueError):
                bootstrap_intervals(actual, predicted, iterations=20, independent_groups=groups)

    def test_bootstrap_catalog_keeps_false_positive_only_class_in_every_resample(self):
        result = bootstrap_intervals(
            [['A'], ['A']], [['A', 'B'], ['A']], iterations=80, seed=11
        )
        self.assertEqual(result['f1_macro']['lower'], .5)
        self.assertEqual(result['f1_macro']['upper'], .5)


if __name__ == '__main__':
    unittest.main()
