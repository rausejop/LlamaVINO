---
name: ejecutar-gguf-en-iris-xe
description: Ejecutar modelos LLAMA en formato GGUF cuantizado sobre una GPU integrada Intel Iris Xe usando OpenVINO GenAI en Windows. Úsalo cuando haya que cargar un .gguf, elegir cuantización/modelo para hardware integrado con poca RAM, o diagnosticar la carga en GPU.
---

# Ejecutar GGUF en Intel Iris Xe con OpenVINO

Conocimiento verificado en este equipo (Windows 11, i7-1165G7, Iris Xe, 16 GB RAM
compartida, Python 3.14, OpenVINO GenAI 2026.2).

## Hechos clave

- La **Iris Xe no tiene VRAM dedicada**: comparte la RAM del sistema. Reserva el
  presupuesto de memoria contando con que Windows ya usa ~5-6 GB.
- OpenVINO la expone como device **`GPU`**; `AUTO` debe preferir `GPU` y caer a `CPU`.
- El **runtime base de OpenVINO no lee GGUF**. Sí lo hace **OpenVINO GenAI**
  (`openvino_genai.LLMPipeline(ruta_gguf, device)`) para arquitecturas soportadas
  (llama, qwen2, phi3, gemma). Si la arquitectura no está soportada o el GGUF no se
  puede leer (p. ej. **Mistral v0.3** da `invalid map<K,T> key`), la carga falla → se
  cae al **motor de respaldo llama.cpp por subproceso** (binario precompilado, no un
  backend de OpenVINO). Ver [[motor-respaldo-llamacpp]].
- Cuantización recomendada: **Q4_K_M** (equilibrio calidad/tamaño y soportada de
  forma fiable). `Q4_0` como alternativa más pequeña; evita `Q8_0`/`Q2`.

## Recomendación de modelo por hardware

| RAM libre real | Modelo | Tamaño aprox. |
| --- | --- | --- |
| ~3-4 GB | Llama 3.2 3B Instruct Q4_K_M | ~2.0 GB (recomendado) |
| muy poca | Llama 3.2 1B Instruct Q4_K_M | ~0.8 GB |
| holgada, sin prisa | Llama 3.1 8B Instruct Q4_K_M | ~4.9 GB (lento en iGPU) |

Para **tool-calling / MCP fiable** hace falta ~7-8B entrenado en function-calling:
**Qwen2.5-7B-Instruct Q4_K_M** (~4.7 GB, arquitectura `qwen2` soportada por OpenVINO
GenAI) es la mejor opción que entra en 16 GB compartidos. Por debajo de 7B el
*tool-calling* es errático. Aun así, en iGPU va lento (~2-4 tok/s).

## Receta mínima

```python
import openvino_genai as ov_genai

pipe = ov_genai.LLMPipeline("modelo-Q4_K_M.gguf", "GPU")  # AUTO→GPU si existe
cfg = ov_genai.GenerationConfig()
cfg.max_new_tokens = 256
historial = ov_genai.ChatHistory()
historial.append({"role": "user", "content": "Hola"})
texto = pipe.generate(historial, cfg, lambda s: (print(s, end=""), False)[1])
```

## Errores y aprendizajes

- **No uses `start_chat()` / `finish_chat()`**: deprecado en OpenVINO GenAI 2026.x.
  Mantén el contexto con `ChatHistory` y pásalo a `generate()` en cada turno.
- **Acentos rotos en consola** (`�`): es la codepage de Windows, no el modelo.
  Reconfigura `sys.stdout`/`sys.stderr` a UTF-8 al arrancar el proceso.
- **Primera carga lenta** (~18-30 s): OpenVINO compila y cachea kernels para la
  Iris Xe; las siguientes ejecuciones arrancan mucho más rápido. Fija un
  `CACHE_DIR` (p. ej. `models/.ovcache`) en la config del plugin para reusarla.
- **`get_state API is supported only when KV-cache compression is disabled`** (GPU):
  el chat con `ChatHistory` necesita `get_state`, incompatible con la compresión u8
  por defecto del KV-cache en GPU. Solución: en la config del plugin de **GPU** pon
  `KV_CACHE_PRECISION = ov.Type.f16` (desactiva la compresión). En CPU no hace falta.
- Rendimiento típico del 1B en esta iGPU: ~1-3 s por respuesta corta.
