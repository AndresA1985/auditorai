<!-- INICIO REPORTE TECNICO TARIFARIO CANDIDATO -->
# Reporte técnico del candidato tarifario prospectivo

## Resultado ejecutivo

La consolidación parte del baseline Git `4e26409`. Se eligió la variante v2
como base de revisión frente a v1 porque incorpora un contrato prospectivo
cerrado: features compartidas entre entrenamiento e inferencia, snapshot
predecisión, partición temporal por grupos conectados, abstención explícita,
procedencia por hashes y gates que impiden publicar métricas clínicas o promover
artefactos sin aprobación independiente. La v1 se conserva únicamente como
referencia comparativa y opción de recuperación; ninguna variante fue activada.

Esta entrega prepara código y contratos. No entrenó modelos, no consumió datos
clínicos reales, no produjo métricas clínicas y no modificó ni reinició el
servicio. Los modelos activos y sus pesos permanecieron intactos.

## Alcance materializado

- Contrato aditivo de evidencia tarifaria con SHA-256 obligatorio cuando existe
  PDF y estados de ausencia sin hash.
- Comparación documental posterior que no cambia códigos, scores, honorarios ni
  anestesia y que conserva el ranking completo incluso sin PDF.
- Prioridad documental `alta` independiente del indicador de revisión; una
  contradicción se expresa como revisión separada, no degradando la prioridad.
- Abstención revisable para toda salida sin códigos.
- `app.features.build_features` como representación común y versionada para las
  ablaciones `clinical`, `document_only` y `clinical_document`.
- Loader de snapshots desidentificados, manifiesto temporal común y purga de
  grupos correlacionados por paciente seudónimo, admisión, hash documental y
  hash exacto de texto clínico normalizado.
- Trainer TF-IDF aislado y trainer Transformer con modelo/tokenizer locales,
  imports diferidos, seeds previos a cargar pesos y salidas nuevas bajo
  `candidate_runs/`.
- Selección de hiperparámetros y umbral únicamente en validation; el test se
  consulta después de congelar la selección. No se hace `final-fit-all`.
- Métricas, bootstrap con catálogo global fijo, cohortes documentales y
  counterfactual sin documento compartidos por ambos trainers.
- Artefactos nacen con `clinical_performance_validated=false`, gate
  `pending_independent_operator_approval`, abstención permitida y promoción
  automática deshabilitada.

## Evidencia de verificación

| Verificación | Resultado | Alcance |
| --- | ---: | --- |
| Suite remota consolidada | 117/117 | Pruebas de contrato y regresión disponibles |
| Contrato Transformer con fakes | 5/5 | Dispatch, orden, metadata y evaluación offline; no es entrenamiento |
| Parseo AST | 37/37 | Todos los archivos Python de `app/`, `scripts/` y `tests/` |
| `git diff --check` | OK | Sin errores de whitespace/conflictos en el diff |
| Entrenamiento ejecutado | 0 | No se abrieron pesos ni se generaron métricas clínicas |
| Datos reales utilizados | 0 | Sólo fixtures sintéticos desidentificados de contrato |
| Promoción/reinicio | 0 | Sin activación, copia a modelos activos ni restart |

Los resultados sintéticos y los fakes sólo validan el pipeline. No prueban
calidad clínica, superioridad frente al baseline ni seguridad para producción.

## Archivos y superficies protegidas

No se cambió la lógica de honorarios, anestesia, servicios, pesos activos ni
configuración operativa. En particular, quedaron fuera del alcance los módulos
de honorarios/anestesia, sus entrenadores, los archivos de modelos activos,
`.env`, `logs/`, systemd y contenedores. Los `__pycache__`/`.pyc` son artefactos
generados y no deben incluirse en el PR o en paquetes de entrega.

La revisión remota confirmó que `aiauditor.service` continuaba activo sobre el
checkout canónico `/home/sistemas201/projects/auditorai`, sin referencias
operativas a las copias candidatas. No hubo restart ni sustitución de modelos.

## Respaldo remoto ya verificado

Directorio de respaldo:

`/home/sistemas201/backups/auditorai_consolidacion_20260921T183642Z`

| Elemento | SHA-256 |
| --- | --- |
| Git bundle | `353cef8cc45aeb39078b41cd87d97249cf9756bd3d970de16eb4251d13966916` |
| Tar candidato v1 | `03fcc186ad210697c931e619b3744a4d9d38140f9ebccfbc77e51e7ea71bba09` |
| Tar candidato v2 | `b3fe5668d8ec6e41213d8cbab70bbe6124b0b83bbdc4e80f0986dcf561f32791` |
| `models_active_SHA256SUMS.txt` | `525eb246e177c89c1057eed2af55f77d624db4f5ac2b0218e112c333c88c4e05` |

Estos hashes identifican respaldos; no son métricas de modelo. Antes de cualquier
recuperación se deben volver a calcular sobre los archivos exactos del respaldo.

## Riesgos y pendientes

- No existe todavía un dataset prospectivo real, revisado e independiente que
  autorice afirmar desempeño clínico.
- El gate de aprobación clínica permanece pendiente y no se puede inferir de un
  F1 sintético, de un fixture ni de un hash válido.
- La integración final con el consumidor de GEN requiere preservar por OR la
  revisión y las discrepancias recibidas, con sus propias pruebas end-to-end.
- No se implementa detección fuzzy de near-duplicates; el agrupamiento documenta
  únicamente coincidencias exactas normalizadas y hashes exactos.
- Se detectó un incidente histórico de empaquetado relacionado con `.env` y
  logs. Este reporte no reproduce su contenido. La rotación de credenciales
  potencialmente afectadas y la purga controlada de copias históricas siguen
  pendientes de autorización y deben tratarse como trabajo de seguridad separado.

## Dictamen

La v2 queda apta para PR y evaluación prospectiva, no para activación clínica.
La aceptación futura exige datos reales revisados, comparación independiente
contra baseline, revisión de etiquetas raras/conflictos/OCR/counterfactual,
aprobación explícita del operador y un cambio operativo separado y reversible.

<!-- FIN REPORTE TECNICO TARIFARIO CANDIDATO -->
