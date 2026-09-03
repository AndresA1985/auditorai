# Contrato de métricas y justificaciones

`codigo_ranking[].score` es un puntaje de ordenamiento no calibrado. No es F1,
confianza ni probabilidad. Cada elemento conserva `codigo`, `score` y `selected`,
y añade `descripcion_codigo`, `texto_soporte` y `justificacion`.

`texto_soporte`, cuando existe, es una cita literal extraída del hallazgo recibido,
es específica para el código y supera el umbral de evidencia del artefacto. La
justificación explica la posición en el ranking, pero no afirma que la cita demuestre
por sí sola la pertinencia clínica. Si no hay soporte suficiente, la justificación
lo declara y solicita validación del auditor.

`prediccion.metricas_modelo` es opcional para mantener compatibilidad con artefactos
anteriores. Solo se publica cuando el artefacto activo contiene `f1_macro` y
`f1_weighted` calculados offline sobre un test holdout. Estas métricas nunca se
calculan a partir del request o registro actual. Son métricas globales del modelo y
nunca se asignan a un código individual.
