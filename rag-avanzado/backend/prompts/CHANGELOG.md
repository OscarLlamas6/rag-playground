# Historial de prompts

Un prompt es código: cada cambio se registra, se prueba y puede revertirse. Los prompts viven en `prompts/<nombre>/<versión>.j2`.

## extraer_tripletas

| Versión | Qué cambió | Por qué |
| --- | --- | --- |
| v1 | Extrae tripletas con tres relaciones permitidas; los nombres salen del texto. | Línea base. La normalización (`graph.py`) lleva cada nombre al oficial y la validación rechaza lo inventado. |
| v2 | Añade la lista de entidades conocidas: nombres exactos y solo esos. | Los nombres libres no se pueden unir en un grafo (el mismo servicio con tres escrituras). Viene del curso. |

## contexto_fragmento

| Versión | Qué cambió | Por qué |
| --- | --- | --- |
| v1 | Recibe el catálogo completo, la sección y el fragmento; devuelve una frase de máximo 30 palabras que lo sitúa. | Contextual Retrieval: la frase se antepone al fragmento antes de vectorizarlo. |

## descomponer

| Versión | Qué cambió | Por qué |
| --- | --- | --- |
| v1 | Separa la pregunta en 1 a 3 subpreguntas de una sola entidad o relación, sin ejemplo resuelto y con salida JSON. | Recuperar cada parte por separado y fusionar con RRF. El curso quitó el ejemplo resuelto porque los modelos pequeños lo copiaban. |

## resumen_comunidad

| Versión | Qué cambió | Por qué |
| --- | --- | --- |
| v1 | Recibe miembros (con cuántos servicios dependen de cada uno), relaciones internas, conexiones externas y equipos; pide 2 o 3 frases. | Resúmenes por comunidad para la búsqueda global. Se incluye el número de dependientes para que el resumen pueda responder «¿cuál es el más usado?». |

## mapa_comunidad

| Versión | Qué cambió | Por qué |
| --- | --- | --- |
| v1 | Extrae del resumen solo lo útil para la pregunta o responde NADA. | Paso «mapa» de la búsqueda global (map-reduce). |

## rag_respuesta

| Versión | Qué cambió | Por qué |
| --- | --- | --- |
| v1 | Contexto numerado con tres tipos (hecho, fragmento, comunidad); cita obligatoria; reglas explícitas para impacto y dependencias; no-sé exacto. | Un modelo pequeño no encadena hechos si no se le dice cómo leer la dirección de «depende de». |
