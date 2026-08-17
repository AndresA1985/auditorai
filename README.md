# auditai

Servicio FastAPI local para sugerir codigos auditor, honorarios por codigo, tiempo de anestesia y nombre de procedimiento desde el historico local de `ap_auditoria_doctor_detalle`.

## Instalacion

```bash
cd /home/anakin/projects/auditorai
/home/virtual/auditoriai/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Edita `.env` con las credenciales y constantes locales antes de ejecutar el servicio.

## Ejecutar

```bash
cd /home/anakin/projects/auditorai
/home/virtual/auditoriai/bin/uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload
```

## Endpoints

- `GET /health`: prueba conexion a base local.
- `GET /clases/codigos`: lista codigos historicos agrupados por frecuencia.
- `POST /predecir_codigos`: recibe datos de una agenda y devuelve propuesta para llenar el modal de Aitrol.

Ejemplo de respuesta esperada por Aitrol:

```json
{
  "ok": true,
  "mensaje": "Propuesta generada. Revise antes de guardar.",
  "prediccion": {
    "codigos": ["43239", "45378"],
    "honorarios_codigo": {"43239": "50", "45378": "100"},
    "honorario": "45378(100),43239(50)",
    "tiempo_anestesia": 4,
    "nombre_procedimiento": "Colonoscopia + EDA",
    "observacion_auditor": "Propuesta generada por IA; validar antes de guardar."
  }
}
```
## Entrenar modelo

El entrenamiento lee ap_auditoria_doctor_detalle desde MySQL, agrupa por agenda auditada y genera un artefacto local en models/auditai_model.joblib.

    cd /home/anakin/projects/auditorai
    /home/virtual/auditoriai/bin/python -m pip install -r requirements.txt
    /home/virtual/auditoriai/bin/python scripts/train_model.py

Para una prueba rapida:

    /home/virtual/auditoriai/bin/python scripts/train_model.py --limit 2000 --output models/auditai_model.joblib

Cuando models/auditai_model.joblib existe, POST /predecir_codigos usa el modelo entrenado. Si no existe, usa el fallback historico anterior.
