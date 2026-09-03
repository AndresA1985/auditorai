"""Entrenador compatible que persiste F1 global del holdout con versión explícita."""

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from sklearn.metrics import f1_score

from scripts import train_multilabel_base as _base
from scripts.train_multilabel_base import *  # noqa: F401,F403

_metricas_originales = _base.metricas_multilabel


def metricas_multilabel(y_true, y_pred) -> dict:
    metricas = _metricas_originales(y_true, y_pred)
    metricas["f1_macro"] = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
    metricas["f1_weighted"] = round(
        float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
    )
    return metricas


_base.metricas_multilabel = metricas_multilabel


def _extraer_argumento(nombre: str) -> str | None:
    if nombre not in sys.argv:
        return None
    indice = sys.argv.index(nombre)
    if indice + 1 >= len(sys.argv):
        raise RuntimeError(f"Falta valor para {nombre}")
    valor = sys.argv[indice + 1]
    del sys.argv[indice:indice + 2]
    return valor


def main() -> None:
    version_modelo = _extraer_argumento("--model-version")
    output = Path(_extraer_argumento("--output") or "models/auditai_multilabel_tfidf_logreg.joblib")
    sys.argv.extend(["--output", str(output)])
    _base.main()
    artefacto = joblib.load(output)
    if version_modelo:
        artefacto["model_version"] = version_modelo
        artefacto["evaluated_at"] = datetime.now().isoformat(timespec="seconds")
        metricas = (artefacto.get("evaluation") or {}).get("final_test_metrics") or {}
        metricas["dataset"] = "test_holdout"
    joblib.dump(artefacto, output)


if __name__ == "__main__":
    main()
