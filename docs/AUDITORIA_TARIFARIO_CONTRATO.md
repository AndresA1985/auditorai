# Integración documental de auditoría tarifaria

Validación realizada en la 201 el 21 de septiembre de 2026. Este cambio formaliza la
integración existente; no entrena, descarga, activa ni promueve modelos. El endpoint
se conserva: `POST /predecir_auditoria`. GEN usa únicamente
`http://192.168.66.33:8008/predecir_auditoria`.

## Contrato de entrada

`PrediccionRequest.evidencia_tarifario` es opcional y contiene únicamente:

| Campo | Contrato |
| --- | --- |
| `estado` | `extraido`, `sin_codigos`, `ilegible`, `sin_documento`, `sin_archivo`, `sin_pdf` |
| `fuente` | `texto`, `ocr`, `mixta`, `ninguna`; corresponde a los orígenes de los códigos |
| `documento_sha256` | SHA-256 hexadecimal de 64 caracteres; obligatorio para PDF presente y ausente cuando no hay PDF |
| `codigos` | Hasta 25 referencias estructuradas |
| `advertencias` | Hasta 25 advertencias del enum `AdvertenciaTarifario` |

Cada referencia contiene `codigo` de 5–8 dígitos, `descripcion` de hasta 180
caracteres, `origen` texto/ocr, `pagina` entera estricta entre 1 y 10 y
`confianza_ocr` opcional entre 0 y 100. Esta última solo corresponde a OCR; no es
un score clínico. No se aceptan campos adicionales en la evidencia ni en sus
referencias: no se transporta PDF, base64 ni texto OCR completo.

Ausencia real (`sin_documento`, `sin_archivo` o evidencia omitida) conserva la
respuesta clínica legada. `sin_pdf` o `pdf_no_adjuntado` indica que se esperaba
documento y requiere revisión. Un PDF sin códigos o ilegible también exige revisión,
sin vaciar una selección clínica existente.

## Comparación posterior al ranking

La anotación conserva exactamente códigos, scores, ranking, selected, honorarios
y anestesia. Cada código documental tiene una opción independiente de prioridad
alta; su score es el original del ranking o `null` si no estaba clasificado. Nunca
se fabrica un score desde la confianza OCR ni se selecciona un código por aparecer
en el PDF. Cualquier OCR, contradicción o discrepancia requiere revisión.

Las citas documentales mantienen página y hash, con `verificada=false` y alcance
`contexto_extraido_gen`: AuditorAI no recibe ni vuelve a abrir los bytes del PDF.
Las contradicciones literales se buscan en cada fuente clínica declarada y conservan
su campo de origen. Una salida clínica sin códigos se declara como abstención,
con revisión y motivo tipado, aunque el PDF tenga referencias.

Se mantiene la compatibilidad del wrapper con artefactos legados y con
`tariff_features_v1`. Esta compatibilidad no busca candidatos ni cambia rutas o
modelos activos. La representación versionada solo se utiliza si el artefacto
configurado la declara explícitamente.

## Pruebas reproducibles

```sh
cd /home/sistemas201/projects/auditorai
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/virtual/autitorai/bin/python -m unittest \
  tests.test_tariff_pdf_contract \
  tests.test_tariff_candidate_review \
  tests.test_tariff_candidate_legacy_wrapper \
  tests.test_tariff_candidate_features -v
```

Resultado: **74 pruebas aprobadas**, sin modelos reales, DB, red ni entrenamiento.
Los tests del wrapper compilan funciones aisladas y usan artefactos sintéticos en
memoria; la prueba de configuración de entrenamiento solo inspecciona AST, nunca
ejecuta `fit`. Las rutas HTTP se prueban por ASGI directo con predictor sustituido.
No se instaló ninguna dependencia ni se descargó ningún modelo.

Además se hicieron tres solicitudes sintéticas al endpoint activo: sin documento,
texto nativo y OCR; las tres respondieron HTTP 200. En las tres se conservaron
exactamente códigos, scores, ranking, selected, honorarios y anestesia. **El proceso
activo todavía responde con el contrato legado y no devuelve los campos
documentales nuevos.** El contrato del código revisado pasa las pruebas aisladas;
su activación y la verificación de la nueva salida HTTP quedan pendientes del
despliegue autorizado. GEN debe conservar su comparación documental de respaldo
para respuestas legadas durante esa transición.

Se completaron otras tres solicitudes sintéticas desde la ruta multipart real de
GEN instalada en una aplicación FastAPI aislada, con ACL y extractor sustituidos
por dobles y **transporte httpx/inferencia reales** al endpoint permitido. Los casos
sin PDF, texto y OCR respondieron HTTP 200: comparación de respaldo correcta,
cada código documental como opción independiente de prioridad alta, revisión
ante divergencia/OCR y trazabilidad con hash/página. Todas las decisiones del
modelo coincidieron exactamente con su respuesta cruda y con el caso sin PDF.
También se verificaron `trust_env=false`, redirects deshabilitados, timeout de
conexión 10 s/lectura 90 s, temporales eliminados y spools cerrados antes del POST,
y `Cache-Control: no-store`. Esta prueba no equivale a reiniciar el proceso GEN.
El reporte restringido `bridge-contract-report.json` conserva solo resultados
booleanos y estados HTTP. Después de estas solicitudes adicionales se repitió
la comparación de los 134 archivos de modelos y de ambos servicios: sin cambios.

## Identidad de modelos y servicios

Manifiesto de modelos (JSON canónico: claves ordenadas y separadores `,` y `:`), SHA-256
`1c2a7fb7061c754d4555385a63caf8bd71dcda7a3e713f0e01ca544da2f67efb`. Los 134 pares de ruta/hash se conservan en el respaldo restringido.

Se compararon 134 hashes de artefactos, pesos y archivos de tokenizador antes y
después de las solicitudes; no cambió ninguno. Artefacto principal configurado:
`models/auditai_transformer_multilabel_v1.4.0.joblib`, SHA-256
`be85317dfb644b0b21841520a27a471b52f0fee090476ddef99e75da9a291190`.

Artefacto de honorarios `models/auditai_honorarios.joblib`, SHA-256
`3c5b212cf74fb1baa7d40616a8f4b5533276d0aba0a14c91016bf2b28744142d`.
Artefacto de anestesia `models/auditai_anestesia.joblib`, SHA-256
`750425e208fd73293c5a3d4a61cd9201dd4b442f8674afe76d0a2260a2ab5644`.

| Servicio | PID conservado | Inicio conservado (UTC-05) |
| --- | --- | --- |
| `aiauditor` | 2656487 | 2026-09-18 06:18:01 |
| `genform` | 3368002 | 2026-09-21 13:36:37 |

La evidencia restringida se encuentra fuera del repositorio en
`/home/sistemas201/tariff-integration-review/20260921T230728Z/auditor-contract/`.
Contiene hashes y resultados sintéticos, sin PDF ni cuerpos clínicos. No debe
agregarse al commit ni a ZIP.

## Despliegue y reversión

1. Revisar el diff y los commits de integración de ambos repositorios; conservar
   todos los cambios ajenos y confirmar que los hashes de modelos no variaron.
2. Ejecutar las pruebas anteriores y la suite completa de GEN en la 201.
3. **Solo después de autorización explícita**, reiniciar `aiauditor` y después
   `genform` en una ventana acordada. No modificar configuración ni rutas de modelos.
4. Repetir los tres casos sintéticos y verificar los campos documentales nuevos,
   prioridad alta, revisión, hash y preservación exacta de decisiones. Verificar
   hashes de modelos y salud de servicios sin consultar logs clínicos.
5. Para revertir, usar `git revert <commit-de-integracion>` sobre la rama desplegada;
   no usar reset destructivo, force-push ni restaurar el worktree completo. Ejecutar
   las pruebas y solicitar autorización explícita antes de reiniciar nuevamente.
   La reversión es de código; los modelos y sus rutas permanecen intactos.
