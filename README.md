# auditai

Servicio FastAPI local para sugerir `codigo_grupo_auditor` desde el historico local de `ap_auditoria_doctor_detalle`.

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

El entrenamiento lee ap_auditoria_doctor_detalle desde MySQL y entrena un modelo enfocado en predecir `codigo_grupo_auditor`. Genera un artefacto local en models/auditai_model.joblib.

    cd /home/anakin/projects/auditorai
    /home/virtual/auditoriai/bin/python -m pip install -r requirements.txt
    /home/virtual/auditoriai/bin/python scripts/train_model.py

Para una prueba rapida:

    /home/virtual/auditoriai/bin/python scripts/train_model.py --limit 2000 --output models/auditai_model.joblib

El entrenamiento evalua con una particion 70/15/15:

- 70% train: casos usados para construir el espacio vectorial y buscar vecinos.
- 15% validation: casos usados para comparar motores, modelos y umbrales.
- 15% test final: casos reservados para reportar la metrica final.

Cuando models/auditai_model.joblib existe, POST /predecir_codigos usa el modelo entrenado. Si no existe, usa el fallback historico anterior.

## Motor neuronal con embeddings

Para entrenar con SentenceTransformer en el servidor 201, agrega estas variables a `.env`:

```env
AUDITORIA_MODEL_ENGINE=embeddings
AUDITORIA_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
AUDITORIA_EMBEDDING_DEVICE=auto
AUDITORIA_EMBEDDING_BATCH_SIZE=64
AUDITORIA_VALIDATION_SIZE=0.15
AUDITORIA_TEST_SIZE=0.15
```

Luego instala dependencias y entrena:

```bash
cd ~/projects/auditorai
pip install -r requirements.txt
python scripts/train_model.py
```

Para comparar contra el baseline anterior:

```bash
python scripts/train_model.py --engine tfidf --output models/auditai_model_tfidf.joblib
```

## Clasificador multi-label

Entrena un baseline supervisado `TF-IDF + LogisticRegression` para predecir codigos individuales de `codigo_grupo_auditor`:

```bash
cd ~/projects/auditorai
python scripts/train_multilabel.py \
  --c-values 0.5,1.0,2.0 \
  --thresholds 0.20,0.30,0.40,0.50 \
  --selection-metric avg_dice \
  --n-jobs 1 \
  --output models/auditai_multilabel_tfidf_logreg.joblib
```

El script usa el split 70/15/15. Selecciona `C` y `threshold` con validation y reporta el resultado final en test con metricas de precision, recall, F1, Dice y overlap.

Para generar un artefacto final entrenado con todos los datos despues de evaluar:

```bash
python scripts/train_multilabel.py --final-fit-all
```

Para usar el artefacto multi-label en el endpoint `/predecir_codigos`, apunta el `.env` al modelo generado:

```env
AUDITORIA_MODEL_PATH=models/auditai_multilabel_tfidf_logreg.joblib
```

El predictor detecta automaticamente si el `.joblib` es `tfidf_logistic_regression_multilabel` y devuelve `codigo_scores` con las probabilidades seleccionadas por codigo.


## Fine-tuning Transformer multi-label

Entrena un encoder Transformer pre-entrenado para clasificacion multi-label de codigo_grupo_auditor.
Entrada: procedimiento_auditor/nombre_procedimiento + hallazgo + conclusion.
Salida: vector multi-label con los codigos historicos disponibles.

Comando sugerido en el servidor 201:

    cd ~/projects/auditorai
    python scripts/train_transformer_multilabel.py --encoder-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --epochs 3 --learning-rate 2e-5 --batch-size 8 --eval-batch-size 16 --thresholds 0.45,0.50,0.55,0.60,0.65 --selection-metric avg_dice --fp16 --output models/auditai_transformer_multilabel.joblib --model-dir models/auditai_transformer_multilabel_hf

El entrenamiento usa BCEWithLogitsLoss, split 70/15/15, y reporta precision, recall, F1, Dice, Jaccard/overlap y exact match.

Para usarlo en /predecir_codigos, apunta el .env al artefacto generado:

    AUDITORIA_MODEL_PATH=models/auditai_transformer_multilabel.joblib

La respuesta del predictor incluye codigo_scores para los codigos seleccionados y codigo_ranking con los mejores scores aunque no pasen el threshold. Ese ranking sirve para diagnosticar casos como 43273 vs 43259.


## Plantillas como referencia estructurada

Las plantillas de auditoria se indexan como conocimiento experto local. Solo se toman items tipo P desde ap_plantilla_items + ap_procedimiento; por defecto se excluye 99204 porque corresponde a consulta y no al procedimiento auditado.

Construir el artefacto de plantillas:

    cd ~/projects/auditorai
    python scripts/build_template_reference.py --output models/auditai_template_reference.joblib --exclude-codes 99204

Cuando models/auditai_template_reference.joblib existe, /predecir_codigos mantiene intactos los codigos del modelo principal y devuelve plantilla_ranking solo como soporte visual para el auditor.

Evaluar offline modelo y cobertura de plantillas:

    cd ~/projects/auditorai
    python scripts/evaluate_hybrid.py --model models/auditai_multilabel_tfidf_logreg.joblib

Probar un encoder medico espanol y compararlo contra el hibrido:

    cd ~/projects/auditorai
    python scripts/train_transformer_multilabel.py --encoder-model PlanTL-GOB-ES/roberta-base-biomedical-clinical-es --epochs 5 --learning-rate 2e-5 --batch-size 8 --eval-batch-size 16 --thresholds 0.20,0.25,0.30,0.35,0.40,0.45,0.50 --selection-metric avg_dice --min-labels 2 --fp16 --output models/auditai_transformer_biomedical_es.joblib --model-dir models/auditai_transformer_biomedical_es_hf

    python scripts/evaluate_hybrid.py --model models/auditai_transformer_biomedical_es.joblib


## Modelo de honorarios por codigo

El modelo de codigos no se cambia. Despues de generar codigo_grupo_auditor, se usa un segundo artefacto para sugerir el porcentaje de honorario por cada codigo generado. Si no existe models/auditai_honorarios.joblib, el sistema conserva el fallback de 100%.

Primer baseline recomendado, mayoria historica por codigo:

    cd ~/projects/auditorai
    python scripts/train_honorarios.py --engine code_majority --final-fit-all --output models/auditai_honorarios.joblib

Prueba alternativa para comparar si el texto aporta informacion adicional:

    cd ~/projects/auditorai
    python scripts/train_honorarios.py --engine tfidf_logreg --class-weight balanced --final-fit-all --output models/auditai_honorarios_tfidf.joblib

El resultado reporta accuracy, macro_f1, weighted_f1, matriz de confusion y reporte por clase para porcentajes 50 y 100.

## Modelo de tiempo de anestesia por codigo

El tiempo de anestesia tambien se predice despues de generar los codigos. La respuesta incluye tiempos_anestesia_codigo con un valor por codigo y conserva tiempo_anestesia como resumen legacy usando el mayor tiempo sugerido.

Primer baseline recomendado, mayoria historica por codigo:

    cd ~/projects/auditorai
    python scripts/train_anestesia.py --engine code_majority --final-fit-all --output models/auditai_anestesia.joblib

Prueba alternativa para comparar si el texto aporta informacion adicional:

    cd ~/projects/auditorai
    python scripts/train_anestesia.py --engine tfidf_logreg --class-weight balanced --final-fit-all --output models/auditai_anestesia_tfidf.joblib

El resultado reporta accuracy, macro_f1, weighted_f1, MAE en horas, matriz de confusion y reporte por clase para tiempos 1,2,3,4,5,6,7,8 y 24.
