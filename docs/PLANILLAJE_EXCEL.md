# Propuestas de planillaje desde el Excel aprobado

La implementación principal pertenece a `/home/mdconsgroup/projects/auditorai`.
`app/predictor.py` conserva el flujo código → plantillas → honorarios → anestesia
y añade `propuesta_planillaje_excel` después de la rama de modelos o del fallback
histórico. Los campos originales permanecen disponibles para comparación; no
son fuente ni completan valores de la propuesta Excel. No existe un campo API
`codigo_auditor`: ese nombre pertenece a la columna AA del libro.

## Fuente y reproducción

Sólo se leen `hallazgos_base` y `HONORARIOS ` (espacio final). La selección usa
trim y rechaza ausencia o ambigüedad. SHA256 del libro aprobado:
`e6766c3acbda7750515158eca1edaa9446b5aca3ea2f42b0b41bb0b79caf13fa`.

`scripts/build_tariff_knowledge.py <libro.xlsx> app/data/tariff_knowledge.json`
se ejecuta con Python bundled y openpyxl, sin escribir ni reexportar el libro.
Procesa las 284 filas, incluidas las 86 sin etiquetas. Se conservan sólo actos
y sitios de vocabulario acotado, códigos, porcentajes, tiempos, incidencias y
coordenadas de origen. No incluye identidades, narrativas completas ni insumos.
No reutiliza normalizadores históricos que concatenan códigos o rellenan 100.
Los valores numéricos con formato `0%`, por ejemplo 1, se interpretan como 100%.
Una lista de porcentajes permanece posicional; una cardinalidad incompleta no
se empareja parcialmente por suposición. Las alternativas con un porcentaje
representan un solo cobro.

El resumen generado cuenta referencias utilizables por la gramática semántica,
no rendimiento clínico. La versión actual puede usar directamente 69 de las 198
filas etiquetadas. Las demás conservan incidencias y pueden aparecer como
referencias parciales. Esto no equivale a validar clínicamente 69 casos ni a
tener cobertura de todas las intervenciones. Una narrativa equivalente que
exceda la gramática puede requerir revisión.

## Cómo se forma la propuesta

`tariff_semantics.py` interpreta actos afirmativos y sitios en hallazgos o en
la descripción 013. El nombre solicitado por sí solo no acredita realización.
La equivalencia compara familias y conjuntos de actos, no nombres exactos.
Se conservan incertidumbres de negación, planes, antecedentes, condiciones,
contradicciones y actos no reconocidos. Las referencias parciales indican los
actos comunes, faltantes y sin correspondencia; nunca autorizan cobros.

`tariff_knowledge.py` devuelve estado `propuesta` o `abstencion`, `lineas`
ordenadas con código/porcentaje/rol/fuentes, `tiempo_anestesia` del conjunto,
`candidatos`, `motivos` y `complementos`. No reparte un tiempo de combinación
entre los códigos, ni lo convierte a minutos. `requiere_revision=true` vive
dentro de esta propuesta; el flag previo de la predicción se conserva.

Ejemplos verificados contra las celdas AA/AB/AC:

- Filas 5/240: polipectomía con pinza y biopsia alta →43250+43239, 100/50, tiempo 2.
- Filas 18/78: canulación biliar, esfinterotomía, colocación de prótesis y soporte
  fluoroscópico →43273+43262+43268+240159, 100/50/100/100, tiempo 4.
- Las79 alternativas 70200003/43239 y70200004/45380 tienen 100% y tiempo 2.
  La evidencia documental puede resolver un código compatible con los actos.
  Sin selección documental inequívoca se presentan las alternativas al auditor.

La posición o un porcentaje 100 no determinan el principal. Se identifica como
adicional sólo lo que HONORARIOS A9/B10 sustenta (43273 y47550); los demás roles
quedan por confirmar. A7 no enumera códigos y no autoriza repartir 50% a todos.
La mención habitual de 240159 en CPRE (A11/B12) no lo añade sin precedente
equivalente y actos que lo sustenten.

## Incidencias de la fuente

No se corrigieron por suposición las filas 112/124 (seis códigos/cinco
porcentajes), 263 (dos códigos/un porcentaje), 216 (último porcentaje 10) ni 188/204
(distribuciones diferentes con los mismos códigos). Se conservan valores y
fuentes y se exige revisión. Los 41 tiempos ausentes permanecen ausentes; no se
promedia ni se completa con otro caso, una regla genérica o un modelo.

## Biopsias013B y sala

Entrada opcional de metadatos de planillaje, ajena a las features de los modelos:

```json
{
  "informes_013b": [{"id_informe": "informe-sintetico-1", "sitios": ["antro", "cuerpo"]}],
  "modalidad_planillaje": "abierto",
  "nivel_sala": "crm_segundo"
}
```

Cada identificador debe corresponder a un informe declarado por el consumidor,
sin duplicados. La API no verifica su existencia en un repositorio documental;
el auditor debe hacerlo. Se cuenta cada informe una vez y se exige concordancia
con sitios biopsiados. Fragmentos, muestras, piezas o sitios no son cantidades
de informes. Un informe con códigos anatómicos incompatibles deja cantidad
pendiente. Los identificadores no se devuelven en la propuesta.

HONORARIOS B14–B17 sustenta 280009, 280017, 280095 y 280025 por sitio o lesión. La
lesión de papila de Vater no se reduce a cualquier papila. Los porcentajes de
esas biopsias quedan nulos porque las hojas no los especifican. Los complementos
son opciones revisables y nunca se suman automáticamente.

Para sala se exige modalidad `abierto` y nivel explícito `crm_segundo` o
`gastro_tercero`. A2/A3 presenta 395162/395272 o 395173/395281 al 100%, respectivamente.
Se ofrecen los derechos para selección del auditor; no se infiere el nivel de
la narrativa ni se agrega automáticamente toda la lista. En paquetes no se
agregan sala ni biopsias separadas. Si la alternativa paquete/abierto está
pendiente, los complementos también quedan pendientes.

## Integración y límites

AuditorAI declara los metadatos y la salida en `schemas.py`; las representaciones
de features siguen usando sus allowlists existentes. El wrapper conserva la
propuesta Excel cuando falta candidato histórico, sin ocultar otros fallos.
No entrena ni promueve pesos, no cambia dependencias ni configuración de servicios.

El puente de formsgenerator conserva ambos formatos PDF y la evidencia local por
celda. Su propuesta documental 70200003/70200004 ahora obtiene 100/2 del Excel y
rechaza actos adicionales o inciertos. Recalcula la propuesta Excel localmente
para usar la asociación documental y metadatos 013B que no entran en los modelos.
Ambos servicios incluyen copias idénticas de los tres módulos compartidos y el importador y
del JSON; cualquier cambio exige regeneración y comprobación de igualdad.

No se implementó interfaz ni persistencia de la decisión del auditor porque no
existen en estos repositorios. El consumidor debe mostrar incertidumbres y
solicitar decisión humana. Las pruebas sintéticas son regresiones técnicas;
no establecen sensibilidad, especificidad ni validación clínica causal.
