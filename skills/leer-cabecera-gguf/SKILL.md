---
name: leer-cabecera-gguf
description: Leer la cabecera de metadatos de un fichero GGUF sin dependencias (arquitectura, nombre, plantilla de chat, tokens), saltando arrays grandes. Úsalo para detectar el motor/arquitectura de un GGUF local, mostrar sus parámetros, o decidir la plantilla de chat sin cargar el modelo.
---

# Leer la cabecera GGUF sin dependencias

Patrón verificado en `gguf_reader.py`. GGUF empieza con un bloque de metadatos
binario que se puede parsear sin librerías ni cargar pesos.

## Formato (little-endian)

1. Magic `GGUF` (4 bytes) + versión (`uint32`) + `tensor_count` (`uint64`) +
   `metadata_kv_count` (`uint64`).
2. Cada KV: `key` (string con longitud `uint64` + bytes UTF-8) + `value_type`
   (`uint32`) + valor según el tipo. El tipo **8 = STRING**; hay enteros, floats,
   bool y **9 = ARRAY** (tipo de elemento + `uint64` longitud + elementos).

## Truco clave: saltar los arrays

Los arrays (vocabulario, merges) son enormes. Para leer **solo escalares** rápido,
`read_scalar_metadata` salta el cuerpo de cada array (`_skip_array`) sin materializarlo.
Esto permite detectar arquitectura/nombre de un modelo de varios GB en milisegundos.

## Para qué se usa

- **Detectar el motor**: `general.architecture` (+ comprobación de nombre
  Mistral/Mixtral) → `OPENVINO_ARCHS = {llama, qwen2, phi3, gemma}` decide OpenVINO vs
  respaldo llama.cpp. Cachea por `(ruta, mtime)`. Ver [[motor-respaldo-llamacpp]].
- **Detectar la plantilla de chat**: `general.name`/`general.architecture` →
  mistral/gemma/phi/qwen/generic.
- **Mostrar parámetros con significado**: `describe_gguf(path, key_filter, array_limit)`
  devuelve filas clave→valor con una explicación en español; `array_limit` capa el
  volcado de arrays (p. ej. tokens).

## Gotchas

- **No leas el fichero entero.** Usa lectura por offsets/streaming; el bloque KV está al
  principio, antes de los tensores.
- Las claves usan punto (`general.name`, `tokenizer.ggml.tokens`); no asumas que existen
  todas: usa `.get` con respaldo al nombre del fichero.
- La lectura puede fallar en GGUF raros; envuelve en try/except y recae en el nombre.
