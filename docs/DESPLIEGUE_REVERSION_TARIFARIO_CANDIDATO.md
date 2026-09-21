<!-- INICIO MANUAL DESPLIEGUE Y REVERSION TARIFARIO CANDIDATO -->
# Manual de despliegue y reversión del candidato tarifario

## Principios obligatorios

Este procedimiento separa tres acciones distintas: sincronizar código, entrenar
un candidato y activar un modelo. Aprobar una no autoriza las siguientes.

- No usar datos, PDF, `.env`, logs ni pesos dentro de ZIP o paquetes de código.
- No ejecutar entrenamiento sobre el checkout productivo ni escribir en
  `models/`; toda salida debe ser una ruta nueva bajo `candidate_runs/`.
- No hacer `final-fit-all`; train, validation y test permanecen separados.
- No promocionar artefactos, cambiar enlaces, reiniciar servicios o retirar
  copias sin una autorización operativa adicional.
- No publicar F1 clínico mientras `clinical_performance_validated` no sea
  exactamente `true` y el gate no registre aprobación independiente.

## 1. Verificación previa del código

Desde la raíz del repositorio:

```bash
git status --short
git diff --check
sha256sum -c docs/CANDIDATO_TARIFARIO_SOURCE_SHA256SUMS.txt
```

Confirmar que el conjunto a revisar no contiene `.env`, PDF, datasets, logs,
pesos, `__pycache__` ni `.pyc`. Confirmar también que no hay cambios en módulos
de honorarios/anestesia ni en sus entrenadores.

## 2. Flujo Git con PR

Publicar únicamente la rama revisada:

```bash
git push origin codex/consolidar-tarifario-prospectivo
```

Abrir un PR hacia la rama destino aprobada. El PR debe incluir revisión del diff,
suite 117/117, AST 37/37, `diff --check`, manifiesto SHA-256 y constancia de que no
hubo entrenamiento ni activación. No mezclar el PR con credenciales, datos o
artefactos de modelos.

Después del merge aprobado, sincronizar el checkout destino sin crear un merge
local implícito:

```bash
git switch <rama-destino-aprobada>
git pull --ff-only origin <rama-destino-aprobada>
git rev-parse HEAD
sha256sum -c docs/CANDIDATO_TARIFARIO_SOURCE_SHA256SUMS.txt
```

Si `pull --ff-only` falla, detenerse y resolver la divergencia mediante revisión;
no usar `reset --hard`, force-push ni sobreescritura del checkout canónico.

## 3. Respaldo antes de evaluar o desplegar

Verificar el respaldo ya comunicado sin extraerlo sobre producción:

`/home/sistemas201/backups/auditorai_consolidacion_20260921T183642Z`

Hashes esperados:

```text
353cef8cc45aeb39078b41cd87d97249cf9756bd3d970de16eb4251d13966916  Git bundle
03fcc186ad210697c931e619b3744a4d9d38140f9ebccfbc77e51e7ea71bba09  tar candidato v1
b3fe5668d8ec6e41213d8cbab70bbe6124b0b83bbdc4e80f0986dcf561f32791  tar candidato v2
525eb246e177c89c1057eed2af55f77d624db4f5ac2b0218e112c333c88c4e05  models_active_SHA256SUMS.txt
```

Usar `sha256sum` contra cada nombre real dentro del directorio y detenerse ante
cualquier diferencia. No mostrar ni copiar `.env` o logs durante la inspección.

## 4. Entrenamiento TF-IDF prospectivo real

El operador debe proporcionar un snapshot clínico real conforme a
`tariff_snapshot_v1`, desidentificado, inmutable y revisado. Para datos reales no
usar `--allow-synthetic`.

```bash
python3 -B scripts/train_tariff_candidate.py \
  --snapshots <snapshot-clinico-revisado.json> \
  --output-dir candidate_runs/<run-nuevo-tfidf> \
  --model-version <version-candidata> \
  --c-values 0.5,1.0,2.0 \
  --thresholds 0.2,0.3,0.4,0.5
```

La ruta de salida debe ser nueva. El comando genera un `split_manifest.json` que
puede reutilizar el Transformer sólo para el mismo hash de dataset y las mismas
fuentes/política del pipeline.

## 5. Entrenamiento Transformer prospectivo real y local

El encoder y tokenizer deben existir localmente y sus rutas absolutas deben ser
aportadas por el operador. El trainer usa `local_files_only=True`; no se permite
descarga implícita.

```bash
python3 -B scripts/train_transformer_tariff_candidate.py \
  --snapshots <snapshot-clinico-revisado.json> \
  --split-manifest candidate_runs/<run-nuevo-tfidf>/split_manifest.json \
  --output-dir candidate_runs/<run-nuevo-transformer> \
  --model-version <version-candidata-transformer> \
  --encoder-model-dir <ruta-absoluta-encoder-local> \
  --tokenizer-dir <ruta-absoluta-tokenizer-local> \
  --thresholds 0.2,0.3,0.4,0.5
```

Los tests con fakes no sustituyen este entrenamiento. Un fake debe permanecer
marcado como simulación, usar claves `simulated_*` y generar cero artefactos.

## 6. Validación de cada run

Revisar, sin mover archivos a `models/`:

1. `run_manifest.json`, `split_manifest.json` y `evaluation_report.json`.
2. Hash de dataset y split; hashes de fuentes, encoder, tokenizer y artefactos.
3. `training_scope=train_split` y ausencia de `final-fit-all`.
4. Selección del umbral en validation y una sola evaluación final del test después
   de congelarlo.
5. Cohortes `document_present`, `document_missing`, `document_empty`, fallos OCR,
   OCR presente y conflicto documento/etiqueta.
6. Counterfactual sin documento y bootstrap por componentes independientes.
7. `clinical_performance_validated=false`, gate pendiente y
   `automatic_promotion=false`.
8. Permisos restrictivos y rutas `model_dir` dentro del run candidato, nunca en
   el directorio activo `models/`.

La aprobación requiere revisión clínica independiente, comparación contra el
baseline y decisión humana documentada. Este manual no autoriza promoción ni
restart. Si finalmente se aprueba una activación, debe tramitarse en otro cambio
con backup inmediato, ventana operativa, health checks y rollback ensayado.

## 7. Reversión de código

La reversión preferida conserva historia:

```bash
git switch -c revert/tarifario-candidato <rama-destino-aprobada>
git revert <commit-de-merge-aprobado>
git push origin revert/tarifario-candidato
```

Abrir y aprobar un PR de reversión. No usar `git reset --hard` sobre el checkout
canónico.

Como recuperación aislada desde el bundle:

```bash
git bundle verify /home/sistemas201/backups/auditorai_consolidacion_20260921T183642Z/<bundle-real>
git clone /home/sistemas201/backups/auditorai_consolidacion_20260921T183642Z/<bundle-real> <directorio-nuevo-de-recuperacion>
```

Verificar primero el SHA-256 del bundle. Clonar en un directorio nuevo; nunca
extraer o clonar encima del checkout activo.

## 8. Recuperación de candidatas y modelos

Para inspeccionar v1/v2, verificar primero sus hashes y listar el tar. Extraer en
un directorio nuevo de recuperación, no sobre el repositorio canónico ni sobre
`models/`. Los tar son recuperación/comparación, no autorización de despliegue.

Para comprobar los modelos activos, verificar el hash del propio
`models_active_SHA256SUMS.txt` y después ejecutar `sha256sum -c` desde la misma
raíz relativa usada al crear el manifiesto. Ante cualquier diferencia, detener
el procedimiento y no reiniciar el servicio.

## 9. Retiro posterior de copias candidatas

Retirar candidatas sólo después de completar todos estos puntos:

1. PR aprobado y checkout canónico sincronizado con `pull --ff-only`.
2. Suite, AST, manifiesto de fuentes y smoke checks satisfactorios.
3. Revalidación de que systemd, procesos, cron, supervisor, PM2, Docker,
   Compose, enlaces y scripts no referencian las rutas candidatas.
4. Bundle, tars y manifiesto de modelos presentes y con hashes correctos.
5. Confirmación de que no hay datos/PDF/`.env`/logs dentro de las candidatas.
6. Autorización explícita para las rutas exactas que se retirarán.

Preferir primero mover cada ruta exacta a cuarentena recuperable con retención
definida. La purga irreversible sólo procede después de esa retención y una nueva
autorización. Nunca aplicar una operación recursiva a `/home`, a la raíz del
workspace, a una variable no resuelta o al checkout canónico.

## 10. Incidente histórico de secretos/logs

No copiar ni abrir su contenido durante este flujo. Tratar las credenciales
potencialmente incluidas en el empaquetado histórico como expuestas: inventariar
sin imprimir valores, rotar en los sistemas propietarios, validar consumidores y
después purgar las copias autorizadas. La rotación y purga siguen pendientes y
deben tener responsables, evidencia y ventana propias.

<!-- FIN MANUAL DESPLIEGUE Y REVERSION TARIFARIO CANDIDATO -->
