from fastapi import FastAPI, Header, HTTPException
import os
import secrets
from .db import fetch_one
from .predictor import clases_codigos, predecir
from .retraining import retraining_status, start_retraining
from .schemas import PrediccionRequest, PrediccionResponse, RetrainingRequest

app = FastAPI(
    title="auditai",
    description="Servicio local para sugerir codigos auditor, honorarios y tiempos de anestesia desde historico de auditoria.",
    version="0.1.0",
)


@app.get("/health")
def health():
    db = fetch_one("SELECT 1 AS ok")
    return {"ok": True, "db": db}


@app.get("/clases/codigos")
def listar_clases_codigos():
    return {"ok": True, "codigos": clases_codigos()}


def authorize_retraining(token: str | None) -> None:
    expected = os.getenv("AUDITORIA_RETRAIN_API_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Reentrenamiento no configurado.")
    if not token or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=403, detail="Reentrenamiento no autorizado.")


@app.post("/admin/retrain", status_code=202)
def request_retraining(
    request: RetrainingRequest,
    x_retrain_token: str | None = Header(default=None),
):
    authorize_retraining(x_retrain_token)
    try:
        status = start_retraining(request.requested_by, request.source_plan_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    return status


@app.get("/admin/retrain/status")
def get_retraining_status(x_retrain_token: str | None = Header(default=None)):
    authorize_retraining(x_retrain_token)
    return retraining_status()


@app.post("/predecir_codigos", response_model=PrediccionResponse)
def predecir_codigos(req: PrediccionRequest):
    try:
        prediccion, score = predecir(req)
        return {
            "ok": True,
            "mensaje": "Propuesta generada. Revise antes de guardar.",
            "prediccion": prediccion,
        }
    except ValueError:
        raise HTTPException(status_code=422, detail="No fue posible generar una predicción con esta entrada.") from None
    except Exception:
        # Exception text may include clinical input, database details or credentials.
        raise HTTPException(status_code=500, detail="Error generando predicción.") from None

@app.post("/predecir_auditoria", response_model=PrediccionResponse)
def predecir_auditoria(req: PrediccionRequest):
    return predecir_codigos(req)
