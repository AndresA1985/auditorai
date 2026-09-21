"""Offline contract tests for the prospective local Transformer trainer."""
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from app.features import build_features
from scripts import train_transformer_tariff_candidate as transformer
from scripts.tariff_snapshot_dataset import SnapshotError


OPTIONS = {
    "epochs": 1.0,
    "learning_rate": 2e-5,
    "batch_size": 2,
    "eval_batch_size": 2,
    "max_length": 64,
    "weight_decay": 0.0,
    "logging_steps": 10,
    "seed": 42,
    "fp16": False,
}


def record(index):
    predicted = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)
    code = "43271" if index % 2 == 0 else "43264"
    token = "CLASE_A" if code == "43271" else "CLASE_B"

    def iso(hours):
        return (predicted + timedelta(hours=hours)).isoformat()

    return {
        "sample_id": f"sample-{index}",
        "patient_group": f"patient-{index}",
        "admission_group": f"admission-{index}",
        "snapshot_immutable": True,
        "source_revision": "prospective_capture",
        "provenance_review_id": "review-QA",
        "label_origin": "auditor_reviewed",
        "prediction_at": iso(0),
        "clinical_snapshot_captured_at": iso(-1),
        "clinical_available_at": iso(-2),
        "label_reviewed_at": iso(24),
        "procedimiento_sistema": f"Procedimiento {token} caso {index}",
        "hallazgos_conclusion": f"Hallazgos {token} prospectivos {index}.",
        "descripcion_estudio_013": f"Estudio {token} prospectivo {index}.",
        "labels": [code],
        "evidencia_tarifario": {
            "estado": "extraido",
            "fuente": "texto",
            "documento_sha256": hashlib.sha256(f"document-{index}".encode()).hexdigest(),
            "codigos": [
                {
                    "codigo": code,
                    "descripcion": f"Referencia {token}",
                    "origen": "texto",
                    "pagina": 1,
                }
            ],
            "advertencias": [],
        },
        "document_available_at": iso(-2),
        "document_snapshot_captured_at": iso(-1),
        "document_hash_verified_at": iso(-0.5),
        "document_same_admission_verified": True,
    }


class RecordingFakeBackend:
    """A deterministic dispatcher. It deliberately is not a training backend."""

    execution_kind = "test_double_dispatch"

    def __init__(self, *, classes, mode, registry):
        self.classes = classes
        self.mode = mode
        self.events = []
        registry.append(self)

    def fit(self, train_texts, train_targets, validation_texts, validation_targets):
        self.events.append(("fit", len(train_texts), len(validation_texts)))
        self.asserted_targets = (train_targets, validation_targets)
        return {"training_performed": False, "backend": "test_double_dispatch"}

    def predict_probabilities(self, texts, *, partition):
        self.events.append(("predict", partition, len(texts)))
        matrix = []
        for text in texts:
            scores = []
            for code in self.classes:
                positive = (
                    (code == "43271" and ("CLASE_A" in text or "43271" in text))
                    or (code == "43264" and ("CLASE_B" in text or "43264" in text))
                )
                scores.append(0.9 if positive else 0.1)
            matrix.append(scores)
        return matrix

    def save(self, destination):
        raise AssertionError("A test double must never emit a trained candidate.")


class TariffTransformerCandidateTests(unittest.TestCase):
    def test_import_is_pure_and_real_loader_is_strictly_local(self):
        source_path = Path(transformer.__file__)
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name.split(".")[0]
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module.split(".")[0]
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module
        )
        self.assertTrue({"torch", "transformers", "numpy", "joblib"}.isdisjoint(imported))
        self.assertEqual(transformer.MODES, ("clinical", "document_only", "clinical_document"))
        self.assertGreaterEqual(source.count("local_files_only=True"), 2)
        main_calls = [
            node for node in tree.body
            if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
        ]
        self.assertEqual(len(main_calls), 1)

    def test_each_backend_sets_seed_before_any_local_from_pretrained_call(self):
        events = []
        fake_numpy = ModuleType("numpy")
        fake_torch = ModuleType("torch")
        fake_torch.utils = SimpleNamespace(data=SimpleNamespace(Dataset=object))
        fake_transformers = ModuleType("transformers")

        class TokenizerLoader:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                events.append("tokenizer_from_pretrained")
                return object()

        class ModelLoader:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                events.append("model_from_pretrained")
                return object()

        fake_transformers.AutoTokenizer = TokenizerLoader
        fake_transformers.AutoModelForSequenceClassification = ModelLoader
        fake_transformers.Trainer = object
        fake_transformers.TrainingArguments = object
        fake_transformers.set_seed = lambda seed: events.append(("set_seed", seed))
        with patch.dict(sys.modules, {
            "numpy": fake_numpy,
            "torch": fake_torch,
            "transformers": fake_transformers,
        }):
            for mode in ("clinical", "document_only"):
                transformer.make_huggingface_backend(
                    encoder_dir=Path("/local/encoder"),
                    tokenizer_dir=Path("/local/tokenizer"),
                    classes=["43264", "43271"],
                    options=OPTIONS,
                    work_dir=Path("/candidate/work") / mode,
                )
        self.assertEqual(events, [
            ("set_seed", OPTIONS["seed"]),
            "tokenizer_from_pretrained",
            "model_from_pretrained",
            ("set_seed", OPTIONS["seed"]),
            "tokenizer_from_pretrained",
            "model_from_pretrained",
        ])

    def test_offline_fake_runs_same_rows_split_and_three_features_without_claiming_training(self):
        rows = [record(index) for index in range(20)]
        with tempfile.TemporaryDirectory(prefix="auditorai-transformer-offline-") as directory:
            root = Path(directory)
            snapshots = root / "snapshots.json"
            snapshots.write_text(
                json.dumps(
                    {
                        "schema_version": "tariff_snapshot_v1",
                        "deidentified": True,
                        "data_kind": "synthetic",
                        "records": rows,
                    }
                ),
                encoding="utf-8",
            )
            encoder = root / "encoder"
            tokenizer = root / "tokenizer"
            encoder.mkdir()
            tokenizer.mkdir()
            (encoder / "config.json").write_text('{"model":"offline-fixture"}', encoding="utf-8")
            (tokenizer / "tokenizer.json").write_text('{"tokenizer":"offline-fixture"}', encoding="utf-8")
            candidate_root = root / "candidate_runs"
            output = candidate_root / "run-001"
            backends = []

            def factory(*, classes, work_dir, **kwargs):
                return RecordingFakeBackend(
                    classes=classes, mode=Path(work_dir).name, registry=backends
                )

            with patch.object(transformer, "build_features", wraps=build_features) as feature_builder:
                report = transformer.run_candidate(
                    snapshots=snapshots,
                    output_dir=output,
                    model_version="transformer-QA-v1",
                    encoder_dir=encoder,
                    tokenizer_dir=tokenizer,
                    thresholds=[0.2, 0.8],
                    options=OPTIONS,
                    allow_synthetic=True,
                    backend_factory=factory,
                    candidate_root=candidate_root,
                    bootstrap_iterations=20,
                )

            self.assertEqual([item["mode"] for item in report["ablations"]], list(transformer.MODES))
            self.assertEqual(len(backends), 3)
            split = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
            expected_sizes = {name: len(split[name]) for name in ("train", "validation", "test")}
            expected_feature_rows = len(rows) + expected_sizes["test"]
            self.assertEqual(feature_builder.call_count, expected_feature_rows * 3)
            called_modes = [call.kwargs["mode"] for call in feature_builder.call_args_list]
            self.assertEqual(
                {mode: called_modes.count(mode) for mode in transformer.MODES},
                {mode: expected_feature_rows for mode in transformer.MODES},
            )
            for backend, ablation in zip(backends, report["ablations"]):
                self.assertEqual(
                    [event[:2] for event in backend.events],
                    [
                        ("fit", expected_sizes["train"]),
                        ("predict", "validation"),
                        ("predict", "test"),
                        ("predict", "missing_document_counterfactual"),
                    ],
                )
                self.assertEqual(ablation["best_params"]["threshold"], 0.8)
                self.assertFalse(ablation["training_performed"])
                self.assertEqual(ablation["status"], "simulated")
                evaluation = ablation["evaluation"]
                self.assertFalse(evaluation["evaluated"])
                self.assertTrue(evaluation["threshold_frozen_before_test"])
                self.assertEqual(evaluation["scope"], "test_double_dispatch_only")
                self.assertNotIn("final_test_metrics", evaluation)
                self.assertIn("simulated_test_metrics", evaluation)
                self.assertNotIn("cohorts", evaluation)
                self.assertNotIn("missing_document_counterfactual", evaluation)
                self.assertEqual(set(evaluation["simulated_cohorts"]), {
                    "document_present", "document_missing", "document_empty",
                    "ocr_failed_or_unreadable", "ocr_present", "document_label_conflict",
                })
                self.assertEqual(
                    evaluation["simulated_missing_document_counterfactual"]["size"],
                    expected_sizes["test"],
                )
            self.assertEqual(report["artifacts"], [])
            self.assertFalse(report["fake_dispatch_is_training"])
            self.assertFalse(report["clinical_performance_validated"])
            self.assertEqual(
                report["clinical_validation_gate"], "pending_independent_operator_approval"
            )
            self.assertEqual(list(output.glob("*.joblib")), [])

            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["arguments"]["modes"], list(transformer.MODES))
            self.assertFalse(manifest["local_sources"]["downloads_allowed"])
            self.assertEqual(manifest["local_sources"]["encoder_sha256"], transformer.tree_digest(encoder))
            self.assertEqual(manifest["local_sources"]["tokenizer_sha256"], transformer.tree_digest(tokenizer))
            self.assertEqual(
                manifest["trainer_source_sha256"],
                transformer.file_digest(Path(transformer.__file__)),
            )
            compatibility = manifest["established_transformer_compatibility"]
            established_source = Path(transformer.__file__).with_name(
                "train_transformer_multilabel_base.py"
            )
            self.assertEqual(
                compatibility["source_path"],
                "scripts/train_transformer_multilabel_base.py",
            )
            self.assertEqual(
                compatibility["source_sha256"], transformer.file_digest(established_source)
            )
            self.assertEqual(
                compatibility["baseline_encoder_id"],
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            )
            self.assertTrue(compatibility["baseline_encoder_id_is_reference_only"])
            self.assertEqual(compatibility["loss"], "BCEWithLogitsLoss")
            self.assertEqual(
                compatibility["architecture"],
                {
                    "model": "AutoModelForSequenceClassification",
                    "tokenizer": "AutoTokenizer",
                },
            )
            self.assertEqual(compatibility["max_length"], OPTIONS["max_length"])
            self.assertEqual(compatibility["selection"]["threshold_candidates"], [0.2, 0.8])
            self.assertIsNone(compatibility["selection"]["selected_threshold"])
            with self.assertRaises(SnapshotError):
                transformer.run_candidate(
                    snapshots=snapshots,
                    output_dir=output,
                    model_version="transformer-QA-v1",
                    encoder_dir=encoder,
                    tokenizer_dir=tokenizer,
                    thresholds=[0.5],
                    options=OPTIONS,
                    allow_synthetic=True,
                    backend_factory=factory,
                    candidate_root=candidate_root,
                    bootstrap_iterations=10,
                )

    def test_local_sources_and_probability_contract_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix="auditorai-transformer-sources-") as directory:
            empty = Path(directory) / "empty"
            empty.mkdir()
            with self.assertRaises(SnapshotError):
                transformer.tree_digest(empty)
        for values in ([[0.5]], [[float("nan"), 0.5]], [[True, 0.5]], [[1.1, 0.5]]):
            with self.subTest(values=values), self.assertRaises(SnapshotError):
                transformer._validate_probabilities(values, 1, 2)

    def test_artifact_metadata_uses_candidate_runtime_path_and_established_contract(self):
        with tempfile.TemporaryDirectory(prefix="auditorai-transformer-artifact-") as directory:
            output = Path(directory) / "candidate_runs" / "run-002"
            metadata = transformer.artifact_runtime_metadata(
                output,
                "clinical_document",
                max_length=384,
                thresholds=[0.25, 0.5],
                selected_threshold=0.5,
            )
            model_dir = Path(metadata["model_dir"])
            active_models = (Path(transformer.__file__).resolve().parents[1] / "models").resolve()
            self.assertTrue(model_dir.is_absolute())
            self.assertIn(output.resolve(), model_dir.parents)
            self.assertNotEqual(model_dir, active_models)
            self.assertNotIn(active_models, model_dir.parents)
            self.assertEqual(metadata["model_dir_relative"], "models/clinical_document")
            compatibility = metadata["established_transformer_compatibility"]
            self.assertEqual(compatibility["max_length"], 384)
            self.assertEqual(compatibility["selection"]["metric"], "f1_macro")
            self.assertEqual(compatibility["selection"]["partition"], "validation")
            self.assertEqual(compatibility["selection"]["selected_threshold"], 0.5)
            self.assertEqual(
                compatibility["selection"]["test_access"], "after_threshold_freeze"
            )
        with self.assertRaises(SnapshotError):
            transformer.artifact_runtime_metadata(
                Path(transformer.__file__).resolve().parents[1],
                "clinical",
                max_length=64,
                thresholds=[0.5],
                selected_threshold=0.5,
            )


if __name__ == "__main__":
    unittest.main()
