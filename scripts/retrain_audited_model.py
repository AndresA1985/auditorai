#!/usr/bin/env python3
"""Retrain from auditor-validated rows and atomically publish the code model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_status(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{0}.{1}.tmp".format(path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o640)
    os.replace(temporary, path)
    fsync_directory(path.parent)


def holdout_dice(artifact: dict) -> float | None:
    value = (
        (artifact.get("evaluation") or {})
        .get("final_test_metrics", {})
        .get("avg_dice")
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and 0 <= value <= 1 else None


def validate_candidate(
    candidate: dict,
    current: dict | None,
    min_rows: int,
    max_dice_regression: float,
) -> dict:
    if not isinstance(candidate, dict):
        raise RuntimeError("El entrenamiento nocturno no produjo un artefacto válido.")
    if candidate.get("model") != "tfidf_logistic_regression_multilabel":
        raise RuntimeError("El artefacto nocturno no corresponde al modelo multi-label esperado.")
    if candidate.get("training_scope") != "all_rows_after_evaluation":
        raise RuntimeError("El artefacto nocturno no fue ajustado con todas las decisiones auditadas.")
    rows = candidate.get("training_rows")
    codes = candidate.get("unique_codes")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < min_rows:
        raise RuntimeError("No existen suficientes decisiones auditadas para publicar el modelo.")
    if isinstance(codes, bool) or not isinstance(codes, int) or codes < 2:
        raise RuntimeError("El modelo nocturno no contiene suficientes códigos distintos.")
    candidate_dice = holdout_dice(candidate)
    if candidate_dice is None:
        raise RuntimeError("El modelo nocturno no contiene una evaluación holdout válida.")

    current_dice = holdout_dice(current or {})
    if current_dice is not None and candidate_dice < current_dice - max_dice_regression:
        raise RuntimeError(
            "El modelo nocturno fue rechazado porque su Dice holdout retrocede más de lo permitido."
        )
    return {
        "training_rows": rows,
        "unique_codes": codes,
        "candidate_avg_dice": candidate_dice,
        "previous_avg_dice": current_dice,
    }


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    import joblib

    sys.path.insert(0, str(ROOT))
    from app.config import settings

    parser = argparse.ArgumentParser(
        description="Reentrena y publica atómicamente el modelo usando decisiones validadas."
    )
    parser.add_argument("--output", default=str(settings.model_path))
    parser.add_argument(
        "--min-rows",
        type=int,
        default=int(os.getenv("AUDITORIA_RETRAIN_MIN_ROWS", "100")),
    )
    parser.add_argument(
        "--max-dice-regression",
        type=float,
        default=float(os.getenv("AUDITORIA_RETRAIN_MAX_DICE_REGRESSION", "0.02")),
    )
    parser.add_argument("--status-file", default="")
    parser.add_argument("--requested-by", default="scheduler")
    parser.add_argument("--source-id", type=int, default=0)
    args = parser.parse_args()
    if args.min_rows < 1 or not 0 <= args.max_dice_regression <= 1:
        raise RuntimeError("Los límites del reentrenamiento nocturno no son válidos.")

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(output.name + ".retrain.lock")
    candidate_path = output.with_name(".{0}.{1}.candidate".format(output.name, os.getpid()))
    status_path = Path(args.status_file) if args.status_file else None
    if status_path is not None and not status_path.is_absolute():
        status_path = ROOT / status_path
    status_base = {
        "requested_by": str(args.requested_by)[:32],
        "source_id": max(0, int(args.source_id)),
        "pid": os.getpid(),
    }

    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            result = {"ok": True, "status": "already_running", **status_base}
            write_status(status_path, {**result, "finished_at": utc_now()})
            print(json.dumps(result))
            return

        write_status(status_path, {
            "ok": True, "status": "running", "started_at": utc_now(), **status_base,
        })
        try:
            command = [
                sys.executable,
                str(ROOT / "scripts" / "train_multilabel.py"),
                "--final-fit-all",
                "--output",
                str(candidate_path),
            ]
            subprocess.run(command, cwd=str(ROOT), check=True)
            candidate = joblib.load(candidate_path)
            current = joblib.load(output) if output.exists() else None
            report = validate_candidate(
                candidate, current, args.min_rows, args.max_dice_regression
            )
            fsync_file(candidate_path)
            previous_mode = stat.S_IMODE(output.stat().st_mode) if output.exists() else 0o640
            os.chmod(candidate_path, previous_mode)
            os.replace(candidate_path, output)
            fsync_directory(output.parent)
            result = {
                "ok": True,
                "status": "published",
                "output": str(output),
                **report,
            }
            write_status(status_path, {
                **result, **status_base, "finished_at": utc_now(),
            })
            print(json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            write_status(status_path, {
                "ok": False,
                "status": "failed",
                "message": str(exc)[:300],
                "finished_at": utc_now(),
                **status_base,
            })
            raise
        finally:
            candidate_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
