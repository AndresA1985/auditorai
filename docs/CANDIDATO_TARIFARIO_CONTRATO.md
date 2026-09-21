# Comparación documental preparada en copia aislada

Esta implementación se prepara en `desarrollo`, a partir de una copia fuente
verificada de AuditorAI. No activa artefactos, no entrena el modelo productivo,
no reinicia servicios y no modifica GEN ni el checkout de producción IECED.
El documento de validación es reciente y **no existe aún un historial real
disponible para entrenamiento supervisado con PDF**. Los fixtures y métricas
sintéticos validan código y reproducibilidad, no mejora o calidad clínica.

## Modelo vigente y comparación posterior

`PrediccionRequest` acepta de forma aditiva `evidencia_tarifario` y
`informe_tecnico_justificacion`. El objeto documental conserva el contrato GEN:
estado, fuente, códigos explícitos, descripción contextual, página, hash y
advertencias. Confianza OCR se refiere únicamente al reconocimiento de caracteres.
No es una probabilidad de procedimiento ni un score del clasificador.

Los artefactos legados conservan sus features, thresholds, mínimo de etiquetas,
códigos seleccionados, score, rankings, honorarios y anestesia. Recibir evidencia
no activa el constructor nuevo. La comparación posterior agrega metadata cuando
existe PDF; una solicitud legada sin él conserva el JSON previo. Los estados
sin_documento/sin_archivo/sin_pdf no añaden revisión ni diferencias por ausencia:
las sugerencias habituales proceden de clínica/hallazgos. El candidato futuro
puede declarar estado ausente y opciones vacías sin elevar revisión por ausencia.
Si cualquier salida deja codigos=[], abstencion=true y requiere_revision=true por
abstención clínica, con motivo_abstencion explícito; esa revisión no se atribuye
a la ausencia de PDF. Sugerencias no vacías conservan el flujo clínico habitual.

`comparacion_ranking_tarifario` describe **cada** candidato de ranking con el
score y selected originales, en_documento, citas y clasificación: coincidencia,
documental_no_seleccionado, adicional o sin_referencia_documental. Un candidato
ranked-only fuera del PDF no crea por sí solo una discrepancia del conjunto elegido.
Las diferencias adicional_modelo/ausente_modelo/sin_evidencia_documental comparan
solo `prediccion.codigos` con los códigos documentales. Nada une ni elimina códigos.

`citas_documentales` y `justificaciones_tarifario` son evidencia independiente
del modelo. Las citas documentales son literalmente codigo o descripcion del
contexto extraído por GEN, con página y hash obligatorios cuando el PDF existe;
no son una cita bruta ni una nueva verificación del PDF. Por ello las citas de
alcance `contexto_extraido_gen` llevan `verificada=false`: el JSON y su hash no
certifican por sí solos causalidad, correspondencia clínica ni revisión visual.
Las citas clínicas deben existir literalmente en su campo original y su alcance
es `clinica_literal`. `verificada` comprueba literalidad, no realización ni verdad
clínica. No se atribuye al modelo legado una justificación documental inexistente.

`opciones_tarifario_documental` presenta cada código PDF explícito con prioridad
alta y etiqueta «Referencia principal del documento». Conserva origen, página,
hash, citas y score OCR separados; modelo_score es el score original de ranking
o None si el código no aparece en él. No fabrica un 99%, no agrega seleccionados
ni cambia prediccion.codigos. coincidencia_modelo indica presencia en ranking;
modelo_selected indica presencia en el conjunto elegido. Una posible negación
explícita conserva prioridad alta y marca `requiere_revision=true` por separado,
con motivo y cita literal.

Toda referencia PDF estructurada exige revisión porque AuditorAI no vuelve a
abrir los bytes del documento. OCR, fallo documental con PDF presente,
divergencia, falta de cita clínica de un seleccionado y posible contradicción
también exigen revisión sin bloquear la propuesta clínica.
Ausencia real de PDF mantiene el flujo clínico habitual. La extensión `contradiccion_clinica`
solo detecta una negación explícita de realización que menciona el código o un
término concreto documental en la misma cláusula de hallazgos; mantiene la cita
literal y no constituye una conclusión clínica. No elimina selecciones.

## Representación futura compartida

`app.features.FEATURE_SCHEMA_VERSION` es `tariff_features_v1`.
`build_features(payload, mode)` es la única representación train/inferencia:

- clinical: clínica e informe técnico disponibles antes de la decisión, sin PDF;
- document_only: códigos/descripciones documentales, sin clínica;
- clinical_document: documento como referencia contextual más clínica íntegra.

No se inventa una cuarta ablación idéntica. Los campos originales del request
permanecen intactos. El constructor retira únicamente el bloque GEN completo y
exacto antes de generar una referencia nueva, evitando duplicación. Excluye
identidad, CV, hash, página, demografía y confianza OCR de features. Descripciones
documentales con directivas se descartan como feature, no se ejecutan.

Solo un artefacto que declare la versión reconocida y feature_mode activa esta
representación. Versiones desconocidas se rechazan. `allow_abstention=true` del
candidato usa min_labels=0; una entrada nueva sin features utilizables se abstiene.
Los artefactos legados no adquieren esta política silenciosamente.

Las métricas API requieren holdout final, training_scope=train_split, evaluación
explícita, valores finitos en [0,1] y tamaño positivo. Artefactos dataset_kind=synthetic
o clinical_performance_validated=false nunca publican F1 como calidad clínica.
Todo artefacto candidato nace con `clinical_performance_validated=false` y gate
`pending_independent_operator_approval`, incluso si el snapshot declara datos
clínicos. Los informes de pipeline sintético o no aprobados permanecen separados.

## Snapshot prospectivo y manifiesto común

`tariff_snapshot_v1` acepta sólo campos predecisión desidentificados, etiquetas
revisadas después de `prediction_at` y evidencia documental disponible y
capturada antes de la predicción. La inmutabilidad se verifica por hash del
archivo completo; los flags JSON son atestaciones que deben auditarse aguas
arriba y no prueban desidentificación ni causalidad.

El split es temporal y conecta transitivamente seudónimo de paciente, admisión,
hash exacto del PDF y hash exacto del texto clínico normalizado. Componentes que
cruzan cortes se purgan. No se implementa detección fuzzy de near-duplicates y
no debe afirmarse lo contrario. El manifiesto registra dataset/split, política,
versiones y hashes de feature builder, splitter y ambos trainers, argumentos
comunes y entorno; cada ejecución agrega un `run_manifest.json` sin contenido
clínico con sus hiperparámetros.

`scripts/train_tariff_candidate.py` ejecuta exactamente tres ablaciones sobre el
mismo manifiesto: `clinical`, `document_only` y `clinical_document`, con TF-IDF
`char_wb(3,5)`, `min_df=2`, `max_features=250000` y LR OvR `liblinear`,
`balanced`, `max_iter=1000`. Nunca activa el artefacto.

`scripts/train_transformer_tariff_candidate.py` consume el mismo loader,
manifiesto y `build_features`. Sus imports pesados y apertura de pesos ocurren
sólo al ejecutar el comando y el encoder debe existir localmente; la integración
no descarga ni entrena pesos.

## Coordinación necesaria antes de activar

GEN vigente vuelve a calcular `requiere_revision` y `discrepancias`; puede
sobrescribir las anotaciones nuevas de AuditorAI. **La comparación candidata
no se presenta como activa end-to-end.** Activar el candidato requiere evaluación
con dataset real prospectivo y coordinar GEN para hacer OR de revisión y preservar
las discrepancias adicionales verificadas cuando hay PDF. GEN actual también
eleva revisión por ausencia: la política final pide retirar solamente esa causa
cuando no existe PDF y mantener la revisión clínica preexistente. Este archivo solo ofrece un fragmento
futuro revisable; no se aplica ni despliega durante esta preparación:

```python
# INICIO CAMBIO FUTURO: conservar revisión/discrepancias de AuditorAI candidato.
upstream_review = bool(output.get("requiere_revision") or prediction.get("requiere_revision"))
document_absent = evidence.get("estado") in {"sin_documento", "sin_archivo", "sin_pdf"} and not evidence.get("codigos")
allowed_types = {"adicional_modelo", "ausente_modelo", "sin_evidencia_documental", "contradiccion_clinica"}
upstream_differences = prediction.get("discrepancias") or output.get("discrepancias") or []
if document_absent:
    differences = []  # ausencia habitual, sin discrepancias ni revisión documentales
for item in upstream_differences:
    if (not document_absent and isinstance(item, dict) and item.get("tipo") in allowed_types
            and isinstance(item.get("codigo"), str) and isinstance(item.get("motivo"), str)
            and len(item["motivo"]) <= 500
            and not any((entry["codigo"], entry["tipo"]) == (item["codigo"], item["tipo"]) for entry in differences)):
        differences.append(item)
review = upstream_review if document_absent else bool(review or upstream_review or differences)
# FIN CAMBIO FUTURO: conservar revisión/discrepancias de AuditorAI candidato.
```

El fragmento necesitará pruebas y validación de citas/contexto en GEN antes de
incorporarlo. No cambia autenticación, evidencia clínica ni autorizaciones.
