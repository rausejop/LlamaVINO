---
name: openvino-ir-cargar-convertir
description: Cargar, convertir e inspeccionar modelos OpenVINO IR (directorio openvino_model.xml/.bin + tokenizer) para LLMs en hardware Intel. Úsalo para cargar un directorio IR con mmap, convertir un modelo de Hugging Face a IR int4, o leer su config/rt_info.
---

# OpenVINO IR: cargar, convertir, inspeccionar

Verificado en LlamaVino (`ir_reader.py`, `crear_motor`, `convert_to_ir`). El IR es la
ruta **nativa de OpenVINO** y a menudo más robusta que GGUF para arquitecturas que el
lector GGUF no ingiere (p. ej. Mistral). Ver [[motor-respaldo-llamacpp]].

## Detectar y cargar un directorio IR

Un IR es un **directorio** con `openvino_model.xml` + `openvino_model.bin` + tokenizer.
`es_modelo_ir(path)` = dir que contiene el `.xml`. Se carga con la misma
`LLMPipeline(dir, device)` que el GGUF, con **mmap** activo (`ov::enable_mmap`, por
defecto), así los pesos se paginan desde el `.bin`.

```python
import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline("./qwen2.5-7b-int4-ov", "GPU")
```

- `mmap` ayuda a la paginación/carga en CPU; en **GPU el modelo aún debe caber** en la
  memoria compartida.
- El contexto sale de `config.json` → `max_position_embeddings`.
- Aplica `KV_CACHE_PRECISION=f16` en GPU (ver [[ejecutar-gguf-en-iris-xe]]).

## Convertir de Hugging Face a IR int4

```powershell
optimum-cli export openvino --model Qwen/Qwen2.5-7B-Instruct `
  --weight-format int4 ./qwen2.5-7b-int4-ov
```

En LlamaVino: `--convert-ir HF_ID --weight-format int4` envuelve este `optimum-cli`.
**Ojo**: bartowski publica **GGUF**, no IR; el IR viene de la org `OpenVINO` en HF
(descarga por `snapshot_download` del directorio completo) o de este conversor.

## Inspeccionar un IR (`describe_ir`)

Lee, **sin cargar el modelo**: ficheros+tamaños, `config.json` (con significados en
español), `generation_config.json` y el `rt_info` del XML (versión de optimum,
compresión de pesos nncf). El `rt_info` se lee del **final** del fichero XML.

**Seguridad XML**: parsea defensivamente — usa `defusedxml` si está; si no, rechaza
DOCTYPE/ENTITY para evitar XXE. Nunca parsees el XML de un IR ajeno con el parser por
defecto sin estas defensas.
