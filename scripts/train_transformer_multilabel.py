"""Entrenador Transformer que liga métricas holdout a una versión explícita."""

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib

from scripts import train_transformer_multilabel_base as _base
from scripts.train_transformer_multilabel_base import *  # noqa: F401,F403

_entrenar_original = _base.entrenar


def entrenar(args):
    resultado = _entrenar_original(args)
    artefacto = joblib.load(args.output)
    version = getattr(args, "model_version", None)
    if version:
        artefacto["model_version"] = version
        artefacto["evaluated_at"] = datetime.now().isoformat(timespec="seconds")
        artefacto["training_scope"] = "train_split"
        metricas = (artefacto.get("evaluation") or {}).get("final_test_metrics") or {}
        metricas["dataset"] = "test_holdout"
        joblib.dump(artefacto, args.output)
    return resultado


_base.entrenar = entrenar


def _extraer_version() -> str | None:
    if "--model-version" not in sys.argv:
        return None
    indice = sys.argv.index("--model-version")
    if indice + 1 >= len(sys.argv):
        raise RuntimeError("Falta valor para --model-version")
    version = sys.argv[indice + 1]
    del sys.argv[indice:indice + 2]
    return version


def main() -> None:
    version = _extraer_version()
    original_parse_args = _base.argparse.ArgumentParser.parse_args

    def parse_args(parser, *args, **kwargs):
        namespace = original_parse_args(parser, *args, **kwargs)
        namespace.model_version = version
        return namespace

    _base.argparse.ArgumentParser.parse_args = parse_args
    _base.main()


if __name__ == "__main__":
    main()
