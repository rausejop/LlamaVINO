---
name: reparto-capas-gpu-cpu
description: Calcular en tiempo real cuántas capas de un LLM van a la GPU (-ngl) vs RAM según la VRAM y RAM libres, para modelos que no caben enteros en una iGPU. Úsalo al planificar el offload de capas, elegir n-gpu-layers, o decidir entre cargar entero vs paginar con mmap.
---

# Reparto de capas GPU/CPU (offload planning)

Verificado en `layer_planner.py`. Calcula, **sin cargar el modelo**, cuántas capas
caben en la GPU y cuántas se quedan en RAM, para no quedarse sin memoria en una iGPU
de memoria compartida (Iris Xe). Devuelve un `OffloadPlan` con `n_gpu_layers`.

## Entradas que lee

- **Tamaño del GGUF** en disco y `block_count` (nº de capas) de la cabecera
  (ver [[leer-cabecera-gguf]]) → coste aproximado de peso por capa.
- **Coste del KV-cache por capa** según contexto (`n_ctx`), nº de cabezas y dimensión.
- **VRAM detectada**: `GPU_DEVICE_TOTAL_MEM_SIZE` del plugin OpenVINO; **RAM libre** del
  sistema.

## Cálculo

```
peso_por_capa ≈ tamaño_modelo / block_count
kv_por_capa   ≈ f(n_ctx, n_kv_heads, head_dim, precisión_kv)
n_gpu_layers  = floor((vram_disponible - reserva) / (peso_por_capa + kv_por_capa))
```

Capa la reserva para el SO/driver; el resto de capas van a CPU/RAM. `--n-gpu-layers
auto|N` deja elegir; `auto` usa el plan.

## Decisión de motor (importante)

- Si el modelo **cabe entero** en VRAM → ruta nativa OpenVINO.
- Si **no cabe** y hay binario llama.cpp → respaldo llama.cpp con **mmap** y `-ngl N` del
  plan (paginación desde disco; arranca aunque no quepa). Ver [[motor-respaldo-llamacpp]].
- `mmap` está activo por defecto en llama.cpp; solo `--no-mmap` si se pide cargar entero.

## Gotchas

- En **iGPU la "VRAM" es RAM compartida**: descuenta lo que ya usa Windows (~5-6 GB) o el
  plan será optimista y el load fallará. Ver [[ejecutar-gguf-en-iris-xe]].
- El KV-cache crece con el contexto: un `n_ctx` grande puede dejar menos capas en GPU que
  el propio peso. Recalcula el plan cuando cambie `n_ctx`.
- Es una **estimación** para arrancar, no exacta; deja margen de reserva.
