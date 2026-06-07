#!/usr/bin/env python
"""Planificador de reparto de capas GPU/CPU para modelos GGUF (LlamaVino).

Cuando un modelo no cabe entero en la VRAM de la GPU, llama.cpp permite descargar
sólo las primeras ``N`` capas a la GPU (``-ngl N``) y dejar el resto en la RAM de
la CPU. Este módulo calcula **en tiempo real** cuántas capas conviene enviar a la
GPU, a partir de:

  * el tamaño del fichero GGUF y su número de capas (``block_count``),
  * el coste de la caché KV por capa (depende del contexto y de la atención),
  * la VRAM total de la GPU (vía OpenVINO) y la RAM libre del sistema.

No carga el modelo: sólo lee la cabecera GGUF (con ``gguf_reader``) y consulta la
memoria disponible. El motor llama.cpp, además, usa ``mmap`` por defecto, de modo
que el modelo se pagina desde disco en lugar de leerse entero en memoria.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import gguf_reader

# Margen de VRAM que se reserva para buffers de cómputo, activaciones, etc.
# (no todo el VRAM sirve para pesos + KV). Conservador para no provocar OOM.
_VRAM_USABLE_FRACTION = 0.85
# Reserva fija adicional para los buffers de cómputo de llama.cpp.
_VRAM_OVERHEAD_BYTES = 384 * 1024 * 1024  # 384 MiB
# Bytes por elemento de la caché KV (llama.cpp usa f16 por defecto).
_KV_BYTES_PER_ELEM = 2


@dataclass
class OffloadPlan:
    """Plan de reparto de capas calculado para un modelo concreto.

    Attributes:
      block_count: Número total de capas transformer del modelo.
      n_gpu_layers: Capas recomendadas a la GPU (``-ngl``). El resto van a la CPU.
      layer_bytes: Bytes de pesos estimados por capa.
      kv_layer_bytes: Bytes de caché KV por capa para el contexto dado.
      per_layer_bytes: Coste total de VRAM por capa (pesos + KV).
      file_size: Tamaño del fichero GGUF en disco.
      vram_total: VRAM total detectada (o None si no hay GPU).
      vram_usable: VRAM utilizable tras aplicar margen y reserva.
      free_ram: RAM libre del sistema (o None si no se pudo medir).
      fits_full_gpu: True si todas las capas caben en la GPU.
      reason: Explicación en español de la decisión.
    """

    block_count: int
    n_gpu_layers: int
    layer_bytes: int
    kv_layer_bytes: int
    per_layer_bytes: int
    file_size: int
    vram_total: int | None
    vram_usable: int | None
    free_ram: int | None
    fits_full_gpu: bool
    reason: str


def gib(num_bytes: float | None) -> str:
    """Formatea bytes como GiB legibles (o ``?`` si es None)."""
    if num_bytes is None:
        return "?"
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def free_ram_bytes() -> int | None:
    """Devuelve la RAM física libre en bytes (o None si no se puede medir)."""
    if os.name == "nt":
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullAvailPhys)
        return None
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 - psutil es opcional
        return None


def vram_total_bytes(device: str = "AUTO") -> int | None:
    """Devuelve la VRAM total de la GPU en bytes vía OpenVINO (o None).

    ``device`` "AUTO" resuelve a la primera GPU enumerada. Para GPUs integradas
    (Iris Xe) este valor es una porción de la RAM compartida.
    """
    try:
        import openvino as ov
    except ImportError:
        return None
    core = ov.Core()
    name = device
    if device.upper() == "AUTO":
        name = next((d for d in core.available_devices if d.startswith("GPU")), None)
    if not name or not name.startswith("GPU"):
        return None
    try:
        return int(core.get_property(name, "GPU_DEVICE_TOTAL_MEM_SIZE"))
    except Exception:  # noqa: BLE001 - la propiedad puede no existir
        return None


def estimate_layer_sizes(metadata: dict, file_size: int, n_ctx: int) -> dict:
    """Estima el coste de VRAM por capa (pesos + caché KV) a partir de la cabecera.

    Args:
      metadata: Metadatos GGUF (de ``gguf_reader.read_gguf_metadata``).
      file_size: Tamaño del fichero GGUF en disco.
      n_ctx: Tamaño de contexto que se usará.

    Returns:
      ``{"block_count", "layer_bytes", "kv_layer_bytes", "per_layer_bytes"}``.
    """
    arch = metadata.get("general.architecture")
    block_count = metadata.get(f"{arch}.block_count") if arch else None
    if not isinstance(block_count, int) or block_count <= 0:
        block_count = 1
    # Pesos por capa: la mayor parte del fichero está en los bloques transformer.
    layer_bytes = int(file_size / block_count)

    # Caché KV por capa: 2 (clave+valor) · n_ctx · dim_kv · bytes_por_elemento.
    embedding = metadata.get(f"{arch}.embedding_length") if arch else None
    heads = metadata.get(f"{arch}.attention.head_count") if arch else None
    kv_heads = metadata.get(f"{arch}.attention.head_count_kv") if arch else None
    if isinstance(embedding, int) and isinstance(heads, int) and heads > 0:
        kv_dim = embedding
        if isinstance(kv_heads, int) and kv_heads > 0:
            kv_dim = int(embedding * kv_heads / heads)  # GQA
        kv_layer_bytes = 2 * n_ctx * kv_dim * _KV_BYTES_PER_ELEM
    else:
        kv_layer_bytes = 0

    return {
        "block_count": block_count,
        "layer_bytes": layer_bytes,
        "kv_layer_bytes": kv_layer_bytes,
        "per_layer_bytes": layer_bytes + kv_layer_bytes,
    }


def plan_offload(
    model_path,
    *,
    device: str = "AUTO",
    n_ctx: int = 4096,
    vram_total: int | None = None,
    free_ram: int | None = None,
    vram_usable_fraction: float = _VRAM_USABLE_FRACTION,
    vram_overhead_bytes: int = _VRAM_OVERHEAD_BYTES,
) -> OffloadPlan:
    """Calcula cuántas capas enviar a la GPU para ``model_path``.

    Los parámetros ``vram_total``/``free_ram`` permiten inyectar valores (para
    pruebas); si son None se miden en tiempo real con OpenVINO y el sistema.

    Returns:
      Un :class:`OffloadPlan` con ``n_gpu_layers`` y el detalle del cálculo.
    """
    file_size = os.path.getsize(model_path)
    metadata = gguf_reader.read_gguf_metadata(model_path)["metadata"]
    sizes = estimate_layer_sizes(metadata, file_size, n_ctx)
    block_count = sizes["block_count"]
    per_layer = max(1, sizes["per_layer_bytes"])

    if vram_total is None:
        vram_total = vram_total_bytes(device)
    if free_ram is None:
        free_ram = free_ram_bytes()

    if not vram_total:
        return OffloadPlan(
            block_count=block_count, n_gpu_layers=0,
            layer_bytes=sizes["layer_bytes"], kv_layer_bytes=sizes["kv_layer_bytes"],
            per_layer_bytes=per_layer, file_size=file_size,
            vram_total=vram_total, vram_usable=None, free_ram=free_ram,
            fits_full_gpu=False,
            reason="No se detectó GPU/VRAM: todas las capas en CPU (RAM, con mmap).",
        )

    vram_usable = int(vram_total * vram_usable_fraction) - vram_overhead_bytes
    vram_usable = max(0, vram_usable)
    cabe = vram_usable // per_layer
    n_gpu_layers = max(0, min(block_count, int(cabe)))
    fits_full = n_gpu_layers >= block_count

    if fits_full:
        reason = (
            f"El modelo cabe entero en la GPU: las {block_count} capas a VRAM "
            f"({gib(block_count * per_layer)} ≈ ≤ {gib(vram_usable)} útiles)."
        )
    elif n_gpu_layers == 0:
        reason = (
            f"No cabe ninguna capa en la VRAM útil ({gib(vram_usable)}); "
            f"todas a CPU (cada capa ≈ {gib(per_layer)})."
        )
    else:
        reason = (
            f"Reparto: {n_gpu_layers} capas a la GPU y "
            f"{block_count - n_gpu_layers} a la CPU. Cada capa ≈ "
            f"{gib(per_layer)} (pesos {gib(sizes['layer_bytes'])} + KV "
            f"{gib(sizes['kv_layer_bytes'])}); VRAM útil {gib(vram_usable)}."
        )

    return OffloadPlan(
        block_count=block_count, n_gpu_layers=n_gpu_layers,
        layer_bytes=sizes["layer_bytes"], kv_layer_bytes=sizes["kv_layer_bytes"],
        per_layer_bytes=per_layer, file_size=file_size,
        vram_total=vram_total, vram_usable=vram_usable, free_ram=free_ram,
        fits_full_gpu=fits_full, reason=reason,
    )
