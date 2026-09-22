# Verificación técnica local · 2026-09-22

Proyecto principal: AuditorAI. El puente formsgenerator conserva su extracción
documental y consume la misma base de reglas. Fuente: exclusivamente las dos
hojas y el SHA256 descritos en `PLANILLAJE_EXCEL.md`.

## Resultados

- Formsgenerator: suite completa 549 aprobadas, 26 omitidas y una advertencia
  deprecada de Starlette/httpx. Luego se añadieron 17 regresiones de los hallazgos
  de revisión, verificadas en la suite del motor compartido.
- AuditorAI: 70 pruebas dirigidas aprobadas (motor Excel, QA independiente,
  serialización HTTP, entrada 013B, aislamiento de features, fallback sin
  candidato histórico y contrato PDF). No se cargaron pesos ni se consultó DB.
- Alternativas documentales: 63 regresiones aprobadas con anestesia 2, conservando
  las decisiones originales de los modelos.
- Cuatro PDFs reales autorizados: códigos sanitizados 43271;
  70200003+70200004;45380;70200003+70200004. Las propuestas clínicas sintéticas
  sobre esa evidencia seleccionan la alternativa correspondiente al 100%/2.
- Revisión senior: resueltos falsos positivos de futuro, intención, ejemplo,
  condicional, negación de confirmación, infinitivos y contradicción posterior.
  Confirmados sitios compuestos 280025, papila 280017 e hígado 280095.

Las pruebas de fidelidad recorren las 284 filas compiladas y comprueban orden,
formato porcentual de Excel, alternativas, ausencias y anomalías sin corregir.
El importador rechaza hojas ambiguas tras trim. Los datos publicados en código
son desidentificados y no contienen narrativas completas, documentos originales,
identidades, credenciales ni archivos de configuración privados.

## Alcance y límites

Estas cifras son comprobaciones técnicas, no evaluación clínica ni estimación
de sensibilidad/especificidad. Las omisiones de la suite general se mantienen;
no se las presenta como aprobadas. No se ejecutó el conjunto completo de pruebas
de entrenamiento/ML de AuditorAI, cuyo runtime pesado no está en el entorno de
verificación usado; los modelos y sus dependencias no se modificaron.

La gramática conserva 69 referencias semánticas utilizables de 198 etiquetadas y
mantiene el resto para revisión o referencias parciales. Hay conflictos de
porcentajes, 41 tiempos ausentes y combinaciones con tiempos diferentes. Se
abstiene ante esas condiciones; no completa con modelos ni catálogos externos.
Los informes 013B son declarados por el consumidor; su existencia requiere
verificación del auditor. No hay interfaz ni persistencia implementada aquí.

## Estado de despliegue

Validado localmente. No publicado a GitHub ni desplegado en servidor 201.
SSH sin contraseña no autenticó; falta acceso interactivo seguro. La revisión
automática rechazó previamente publicar código privado en GitHub sin
autorización específica de destino/rama. Ese permiso sigue pendiente.
El commit anterior 77b972a no debe desplegarse porque contiene la regla obsoleta.

Los dos servicios requieren código nuevo, preservando cambios ajenos de
configuración, accesos y secretos. Terra debe ejecutar preflight reciente,
respaldo reversible, comparación de hashes de modelos y comprobación de salud.
No se modificaron pesos, dependencias, configuración de servicios ni producción
durante esta verificación local.
