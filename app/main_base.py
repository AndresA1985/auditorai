from fastapi import FastAPI, HTTPException
from .db import fetch_one
from .predictor import clases_codigos, predecir
from .schemas import PrediccionRequest, PrediccionResponse

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
