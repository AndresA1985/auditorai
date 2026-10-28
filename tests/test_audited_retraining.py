import os
from pathlib import Path
import unittest


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


from scripts.retrain_audited_model import validate_candidate


def artifact(rows=200, dice=0.72):
    return {
        "model": "tfidf_logistic_regression_multilabel",
        "training_scope": "all_rows_after_evaluation",
        "training_rows": rows,
        "unique_codes": 12,
        "evaluation": {"final_test_metrics": {"avg_dice": dice}},
    }


class AuditedRetrainingTests(unittest.TestCase):
    def test_training_and_fallback_read_only_auditor_validated_rows(self):
        root = Path(__file__).resolve().parents[1]
        sources = [
            root / "scripts" / "train_model.py",
            root / "scripts" / "train_honorarios.py",
            root / "scripts" / "train_anestesia.py",
            root / "app" / "predictor.py",
        ]
        for source in sources:
            content = source.read_text()
            self.assertIn("estado = 2", content, str(source))
            self.assertNotIn("estado = 1", content, str(source))

        training_source = sources[0].read_text()
        self.assertIn("id_hc_procedimientos", training_source)

    def test_candidate_must_not_regress_beyond_the_configured_gate(self):
        report = validate_candidate(artifact(dice=0.71), artifact(dice=0.72), 100, 0.02)
        self.assertEqual(200, report["training_rows"])

        with self.assertRaisesRegex(RuntimeError, "Dice holdout"):
            validate_candidate(artifact(dice=0.60), artifact(dice=0.72), 100, 0.02)

    def test_candidate_requires_enough_audited_rows(self):
        with self.assertRaisesRegex(RuntimeError, "suficientes decisiones auditadas"):
            validate_candidate(artifact(rows=20), None, 100, 0.02)

    def test_systemd_timer_is_nightly_and_persistent(self):
        root = Path(__file__).resolve().parents[1]
        timer = (root
            / "deploy" / "auditorai-retrain.timer").read_text()
        self.assertIn("OnCalendar=*-*-* 02:15:00", timer)
        self.assertIn("Persistent=true", timer)

        runner = (root / "scripts" / "retrain_audited_model.py").read_text()
        self.assertIn("fcntl.LOCK_EX | fcntl.LOCK_NB", runner)
        self.assertIn("os.replace(candidate_path, output)", runner)

    def test_authenticated_api_starts_a_non_blocking_retraining_process(self):
        root = Path(__file__).resolve().parents[1]
        api = (root / "app" / "main_base.py").read_text()
        orchestration = (root / "app" / "retraining.py").read_text()
        runner = (root / "scripts" / "retrain_audited_model.py").read_text()

        self.assertIn('@app.post("/admin/retrain", status_code=202)', api)
        self.assertIn('@app.get("/admin/retrain/status")', api)
        self.assertIn("secrets.compare_digest", api)
        self.assertIn("start_new_session=True", orchestration)
        self.assertIn('"--status-file"', orchestration)
        self.assertIn('"status": "published"', runner)


if __name__ == "__main__":
    unittest.main()
