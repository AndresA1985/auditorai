"""Authenticated, non-blocking audited-model retraining orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
STATUS_PATH = ROOT / "logs" / "audited_retrain_status.json"
LOG_PATH = ROOT / "logs" / "audited_retrain_api.log"
ENV_PATH = Path(os.getenv("AUDITORIA_RETRAIN_ENV_FILE", "/etc/auditorai-retrain.env"))
ALLOWED_ENV_KEYS = {
    "DB_HOST", "DB_PORT", "DB_DATABASE", "DB_USERNAME", "DB_PASSWORD",
    "AUDITORIA_RETRAIN_MIN_ROWS", "AUDITORIA_RETRAIN_MAX_DICE_REGRESSION",
}
PUBLIC_STATUS_KEYS = {
    "ok", "status", "requested_by", "source_id", "pid", "started_at",
    "finished_at", "training_rows", "unique_codes", "candidate_avg_dice",
    "previous_avg_dice", "message",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_environment() -> dict[str, str]:
    if not ENV_PATH.is_file():
        raise RuntimeError("El entorno protegido de reentrenamiento no está configurado.")
    values: dict[str, str] = {}
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in ALLOWED_ENV_KEYS:
            values[key] = value.strip().strip('"').strip("'")
    if not {"DB_HOST", "DB_PORT", "DB_DATABASE", "DB_USERNAME", "DB_PASSWORD"} <= values.keys():
        raise RuntimeError("El entorno protegido de reentrenamiento está incompleto.")
    return values


def _write_status(payload: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_PATH.with_name(".{0}.{1}.tmp".format(STATUS_PATH.name, os.getpid()))
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.chmod(temporary, 0o640)
    os.replace(temporary, STATUS_PATH)


def _pid_running(pid: int) -> bool:
    if pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def retraining_status() -> dict:
    if not STATUS_PATH.is_file():
        return {"ok": True, "status": "never_run"}
    try:
        payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"ok": False, "status": "unknown"}
    if not isinstance(payload, dict):
        return {"ok": False, "status": "unknown"}
    if payload.get("status") in {"queued", "running"}:
        try:
            pid = int(payload.get("pid") or 0)
        except (TypeError, ValueError):
            pid = 0
        if not _pid_running(pid):
            payload.update({
                "ok": False,
                "status": "failed",
                "message": "El proceso de reentrenamiento terminó sin publicar un resultado.",
                "finished_at": utc_now(),
            })
            _write_status(payload)
    return {key: payload[key] for key in PUBLIC_STATUS_KEYS if key in payload}


def start_retraining(requested_by: str, source_id: int) -> dict:
    current = retraining_status()
    if current.get("status") in {"queued", "running"}:
        return current

    environment = os.environ.copy()
    environment.update(_read_environment())
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_handle = LOG_PATH.open("ab")
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "scripts" / "retrain_audited_model.py"),
                "--status-file", str(STATUS_PATH),
                "--requested-by", requested_by,
                "--source-id", str(source_id),
            ],
            cwd=str(ROOT),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()

    queued = {
        "ok": True,
        "status": "queued",
        "requested_by": requested_by[:32],
        "source_id": source_id,
        "pid": process.pid,
        "started_at": utc_now(),
    }
    _write_status(queued)
    return queued
