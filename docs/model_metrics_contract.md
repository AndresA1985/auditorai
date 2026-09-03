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

## Tres evidencias por codigo

Cada elemento de `prediccion.codigo_ranking` puede incluir el objeto aditivo
`evidencias`. Se mantienen sin cambios los campos planos `texto_soporte`,
`justificacion`, `score`, los mapas `honorarios_codigo` y
`tiempos_anestesia_codigo`, para que los consumidores anteriores sigan
funcionando.

```json
{
  "codigo": "45380",
  "score": 0.1351,
  "evidencias": {
    "codigo": {
      "disponible": true,
      "valor_sugerido": "45380",
      "score_ranking": 0.1351,
      "porcentaje_ranking": 13.51,
      "texto_soporte": "SE TOMAN BIOPSIAS ESCALONADAS DE COLON.",
      "justificacion": "...",
      "fuente": "modelo_codigos"
    },
    "honorario": {
      "disponible": true,
      "valor_sugerido": "100",
      "unidad": "porcentaje_honorario",
      "score_ranking": 0.8,
      "porcentaje_ranking": 80.0,
      "texto_soporte": "SE TOMAN BIOPSIAS ESCALONADAS DE COLON.",
      "justificacion": "...",
      "fuente": "modelo_honorarios"
    },
    "tiempo_anestesia": {
      "disponible": true,
      "valor_sugerido": "2",
      "unidad": "horas",
      "score_ranking": 0.75,
      "porcentaje_ranking": 75.0,
      "texto_soporte": "SE INTRODUCE VIDEOCOLONOSCOPIO HASTA CIEGO.",
      "justificacion": "...",
      "fuente": "modelo_tiempo_anestesia_transformer"
    }
  }
}
```

Las tres evidencias son independientes. `score_ranking` esta expresado entre
0 y 1 y `porcentaje_ranking` es exactamente el mismo puntaje multiplicado por
100 para facilitar su visualizacion. Ninguno de los dos es F1, confianza
clinica ni probabilidad calibrada.

`valor_sugerido` es la salida elegida por cada modelo y no debe confundirse
con `porcentaje_ranking`. Por ejemplo, un honorario con
`valor_sugerido: "100"` significa un honorario sugerido de 100%, mientras
`porcentaje_ranking: 80.0` indica que esa clase obtuvo un puntaje relativo de
80% dentro del modelo de honorarios.

`score_ranking` y `porcentaje_ranking` se calculan usando el texto completo
de la solicitud. Por separado, se puntua cada fragmento literal para la clase
sugerida y se considera evidencia unicamente al fragmento mas relacionado que
iguale o supere `artifact.evidence_threshold`; el valor conservador por defecto
es 0.5. El score del fragmento no reemplaza el score de ranking del texto
completo. Si ningun fragmento supera el umbral, `texto_soporte` es `null` y
la justificacion declara que no existe evidencia textual especifica suficiente.
Si el modelo activo no permite calcular un puntaje textual independiente,
`disponible` es `false`, los puntajes y el texto son `null`, y la
justificacion lo declara expresamente. Todos los resultados requieren
validacion del auditor.
