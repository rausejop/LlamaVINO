#!/usr/bin/env python
"""LlamaVino - Ejecuta un modelo GGUF LLAMA cuantizado con OpenVINO en Intel Iris Xe.

Autor: (c) Rafael Ausejo Prieto, con ayuda de Claude Code.

Este script carga un modelo GGUF de llama.cpp y genera texto a través del runtime
OpenVINO GenAI, prefiriendo la GPU integrada de Intel (Iris Xe) cuando está
disponible y recurriendo a la CPU en caso contrario.

Nota GGUF -> OpenVINO
---------------------
El runtime base de OpenVINO no lee GGUF directamente. OpenVINO GenAI (>= 2025.1)
añadió la carga nativa de GGUF para un subconjunto de arquitecturas (llama,
qwen2, ...): la pipeline ingiere el GGUF, lo descuantiza/convierte en memoria y lo
ejecuta en el dispositivo elegido. Esto reproduce las optimizaciones del backend
OpenVINO de llama.cpp (https://github.com/ggml-org/llama.cpp) manteniéndolo todo
en Python.

Instalación:
    python -m pip install --upgrade openvino openvino-genai huggingface_hub

Uso:
    python LlamaVino.py                              # selector interactivo + chat
    python LlamaVino.py -i                           # fuerza el modo interactivo
    python LlamaVino.py --download llama-3.2-3b      # descarga un GGUF recomendado
    python LlamaVino.py --model model-q4_k_m.gguf --prompt "Hola, ¿quién eres?"
    python LlamaVino.py -m model.gguf -p "..." --device CPU --max-new-tokens 256
    python LlamaVino.py --list-devices
    python LlamaVino.py --list-models
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import code_structure
import edit_formats
import gguf_reader
import ir_reader
import layer_planner
import llama_engine
import mcp_client
import perf_db

__version__ = "1.0.0"

# Modelos GGUF recomendados, dimensionados para una Intel Iris Xe (RAM compartida).
# Cada entrada asocia un alias corto a un repo + fichero de Hugging Face. Q4_K_M es
# la cuantización que OpenVINO GenAI lee de forma fiable para la arquitectura llama.
# Modelos GGUF recomendados (repos de bartowski), dimensionados para una Intel
# Iris Xe (RAM compartida, ~16 GB). Ordenados de MÁS a MENOS potente. Q4_K_M es la
# cuantización equilibrada. ``engine`` marca el motor: "openvino" (nativo:
# arquitecturas llama/qwen2/phi3) o "llamacpp" (respaldo, p. ej. mistral/gemma3).
# Se omiten los > ~14B (p. ej. Qwen2.5-32B/72B, Llama-3.3-70B): no caben en 16 GB.
MODELS: dict[str, dict[str, str]] = {
    "unsloth-qwen2.5-coder-14b": {
        "repo": "unsloth/Qwen2.5-Coder-14B-Instruct-GGUF",
        "file": "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
        "size": "~9.0 GB",
        "note": "unsloth Qwen2.5-Coder 14B (arq. qwen2): el unsloth más potente que "
                "corre por OpenVINO en esta máquina.",
        "engine": "openvino",
        "highlight": True,
    },
    "qwen2.5-14b": {
        "repo": "bartowski/Qwen2.5-14B-Instruct-GGUF",
        "file": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        "size": "~9.0 GB",
        "note": "Qwen2.5 14B: el más potente que cabe; nativo OpenVINO. Lento en iGPU.",
        "engine": "openvino",
    },
    "phi-4": {
        "repo": "bartowski/phi-4-GGUF",
        "file": "phi-4-Q4_K_M.gguf",
        "size": "~9.1 GB",
        "note": "Phi-4 14B (arq. phi3): el Phi más potente; nativo OpenVINO. Pesado.",
        "engine": "openvino",
    },
    "qwen2.5-coder-14b": {
        "repo": "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF",
        "file": "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
        "size": "~9.0 GB",
        "note": "Qwen2.5-Coder 14B: el mejor para programar; nativo OpenVINO.",
        "engine": "openvino",
    },
    "phi-3-medium": {
        "repo": "bartowski/Phi-3-medium-4k-instruct-GGUF",
        "file": "Phi-3-medium-4k-instruct-Q4_K_M.gguf",
        "size": "~8.6 GB",
        "note": "Phi-3 medium 14B (arq. phi3): nativo OpenVINO. Pesado.",
        "engine": "openvino",
    },
    "llama-3.1-8b": {
        "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "size": "~4.9 GB",
        "note": "Llama 3.1 8B: capaz y equilibrado; nativo OpenVINO.",
        "engine": "openvino",
    },
    "qwen2.5-7b": {
        "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "size": "~4.7 GB",
        "note": "Qwen2.5 7B: tool-calling fiable (arq. qwen2); nativo OpenVINO.",
        "engine": "openvino",
    },
    "mistral-7b": {
        "repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "file": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "size": "~4.4 GB",
        "note": "Mistral v0.3: OpenVINO no lo lee; respaldo llama.cpp (plantilla Mistral).",
        "engine": "llamacpp",
    },
    "qwen2.5-coder-7b": {
        "repo": "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        "file": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        "size": "~4.7 GB",
        "note": "Qwen2.5-Coder 7B: programar con menos memoria; nativo OpenVINO.",
        "engine": "openvino",
    },
    "gemma-3-4b": {
        "repo": "bartowski/google_gemma-3-4b-it-GGUF",
        "file": "google_gemma-3-4b-it-Q4_K_M.gguf",
        "size": "~2.5 GB",
        "note": "Gemma 3 4B: buen equilibrio; respaldo llama.cpp (plantilla Gemma).",
        "engine": "llamacpp",
    },
    "phi-3.5-mini": {
        "repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "file": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "size": "~2.4 GB",
        "note": "Phi-3.5 mini 3.8B (arq. phi3): pequeño y rápido; nativo OpenVINO.",
        "engine": "openvino",
    },
    "llama-3.2-3b": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size": "~2.0 GB",
        "note": "Llama 3.2 3B: rápido y solvente; nativo OpenVINO.",
        "engine": "openvino",
    },
    "qwen2.5-3b": {
        "repo": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "file": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "size": "~2.0 GB",
        "note": "Qwen2.5 3B: ligero (arq. qwen2); nativo OpenVINO.",
        "engine": "openvino",
    },
    "gemma-3-1b": {
        "repo": "bartowski/google_gemma-3-1b-it-GGUF",
        "file": "google_gemma-3-1b-it-Q4_K_M.gguf",
        "size": "~0.8 GB",
        "note": "Gemma 3 1B: minúsculo y rápido; respaldo llama.cpp (plantilla Gemma).",
        "engine": "llamacpp",
    },
    "llama-3.2-1b": {
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "size": "~0.8 GB",
        "note": "Llama 3.2 1B: el más rápido; ideal para probar. Nativo OpenVINO.",
        "engine": "openvino",
    },
    # --- Modelos en formato OpenVINO IR (repos REALES de la org OpenVINO en HF,
    # verificados vía la API de Hugging Face). Se descargan como repo completo
    # (snapshot) a un directorio y se cargan con mmap. ``format: "ir"`` los
    # distingue de los GGUF (``dir`` en vez de ``file``). Ordenados por potencia;
    # sólo INT4 que caben en ~16 GB.
    "qwen2.5-14b-ir": {
        "repo": "OpenVINO/Qwen2.5-14B-Instruct-int4-ov",
        "dir": "Qwen2.5-14B-Instruct-int4-ov",
        "size": "~8.5 GB",
        "note": "Qwen2.5 14B en IR INT4 (oficial OpenVINO); el más potente que cabe.",
        "engine": "openvino", "format": "ir",
    },
    "phi-4-ir": {
        "repo": "OpenVINO/phi-4-int4-ov",
        "dir": "phi-4-int4-ov",
        "size": "~8.5 GB",
        "note": "Phi-4 14B en IR INT4 (oficial OpenVINO).",
        "engine": "openvino", "format": "ir",
    },
    "qwen3-14b-ir": {
        "repo": "OpenVINO/Qwen3-14B-int4-ov",
        "dir": "Qwen3-14B-int4-ov",
        "size": "~8.5 GB",
        "note": "Qwen3 14B en IR INT4 (oficial OpenVINO); generación reciente.",
        "engine": "openvino", "format": "ir",
    },
    "deepseek-r1-14b-ir": {
        "repo": "OpenVINO/DeepSeek-R1-Distill-Qwen-14B-int4-ov",
        "dir": "DeepSeek-R1-Distill-Qwen-14B-int4-ov",
        "size": "~8.5 GB",
        "note": "DeepSeek-R1 destilado 14B en IR INT4 (oficial OpenVINO); razonamiento.",
        "engine": "openvino", "format": "ir",
    },
    "qwen2.5-coder-14b-ir": {
        "repo": "OpenVINO/Qwen2.5-Coder-14B-Instruct-int4-ov",
        "dir": "Qwen2.5-Coder-14B-Instruct-int4-ov",
        "size": "~8.5 GB",
        "note": "Qwen2.5-Coder 14B en IR INT4 (oficial OpenVINO); el mejor para programar.",
        "engine": "openvino", "format": "ir",
    },
    "phi-3-medium-ir": {
        "repo": "OpenVINO/Phi-3-medium-4k-instruct-int4-ov",
        "dir": "Phi-3-medium-4k-instruct-int4-ov",
        "size": "~8 GB",
        "note": "Phi-3 medium 14B en IR INT4 (oficial OpenVINO).",
        "engine": "openvino", "format": "ir",
    },
    "gemma-3-12b-ir": {
        "repo": "OpenVINO/gemma-3-12b-it-int4-ov",
        "dir": "gemma-3-12b-it-int4-ov",
        "size": "~7 GB",
        "note": "Gemma 3 12B en IR INT4 (oficial OpenVINO).",
        "engine": "openvino", "format": "ir",
    },
    "gemma-2-9b-ir": {
        "repo": "OpenVINO/gemma-2-9b-it-int4-ov",
        "dir": "gemma-2-9b-it-int4-ov",
        "size": "~5.5 GB",
        "note": "Gemma 2 9B en IR INT4 (oficial OpenVINO).",
        "engine": "openvino", "format": "ir",
    },
    "qwen3-8b-ir": {
        "repo": "OpenVINO/Qwen3-8B-int4-ov",
        "dir": "Qwen3-8B-int4-ov",
        "size": "~5 GB",
        "note": "Qwen3 8B en IR INT4 (oficial OpenVINO).",
        "engine": "openvino", "format": "ir",
    },
    "qwen2.5-7b-ir": {
        "repo": "OpenVINO/Qwen2.5-7B-Instruct-int4-ov",
        "dir": "Qwen2.5-7B-Instruct-int4-ov",
        "size": "~4.7 GB",
        "note": "Qwen2.5 7B en IR INT4 (oficial OpenVINO): mejor relación "
                "potencia/velocidad (mmap + caché). RECOMENDADO.",
        "engine": "openvino", "format": "ir",
    },
    "mistral-7b-ir": {
        "repo": "OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov",
        "dir": "Mistral-7B-Instruct-v0.3-int4-ov",
        "size": "~4 GB",
        "note": "Mistral 7B v0.3 en IR INT4 (oficial OpenVINO): Mistral SÍ por OpenVINO.",
        "engine": "openvino", "format": "ir",
    },
    "gemma-3-4b-ir": {
        "repo": "OpenVINO/gemma-3-4b-it-int4-ov",
        "dir": "gemma-3-4b-it-int4-ov",
        "size": "~2.5 GB",
        "note": "Gemma 3 4B en IR INT4 (oficial OpenVINO): Gemma 3 SÍ por OpenVINO.",
        "engine": "openvino", "format": "ir",
    },
    "gemma-4-e4b-ir": {
        "repo": "OpenVINO/gemma-4-E4B-it-int4-ov",
        "dir": "gemma-4-E4B-it-int4-ov",
        "size": "~3 GB",
        "note": "Gemma 4 E4B en IR INT4 (oficial OpenVINO); generación más reciente.",
        "engine": "openvino", "format": "ir",
    },
    "phi-3.5-mini-ir": {
        "repo": "OpenVINO/Phi-3.5-mini-instruct-int4-ov",
        "dir": "Phi-3.5-mini-instruct-int4-ov",
        "size": "~2.3 GB",
        "note": "Phi-3.5 mini 3.8B en IR INT4 (oficial OpenVINO); ligero.",
        "engine": "openvino", "format": "ir",
    },
    "tinyllama-ir": {
        "repo": "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov",
        "dir": "TinyLlama-1.1B-Chat-v1.0-int4-ov",
        "size": "~0.7 GB",
        "note": "TinyLlama 1.1B en IR INT4 (oficial OpenVINO); el más ligero.",
        "engine": "openvino", "format": "ir",
    },
}

# Arquitecturas GGUF que OpenVINO GenAI carga de forma nativa; el resto (o los
# marcados como "llamacpp" en MODELS) usan el respaldo de llama.cpp.
OPENVINO_ARCHS = {"llama", "qwen2", "phi3", "gemma"}

# Prioridad de recomendación para esta máquina (Iris Xe, 16 GB). Menor = mejor.
# 1) mejor relación potencia/velocidad · 4–9) potentes pero lentos (14B) ·
# 10–13) los más rápidos. El menú muestra esta prioridad y ordena por ella.
RECOMENDACION = {
    "qwen2.5-7b-ir": 1,                 # 7B IR INT4: el mejor equilibrio
    "qwen2.5-7b": 2,                    # 7B GGUF nativo
    "llama-3.1-8b": 3,                  # 8B, alternativa equilibrada
    "qwen2.5-14b-ir": 4,               # 14B IR: más potente, más lento
    "qwen2.5-14b": 5,
    "phi-4-ir": 6,
    "phi-4": 7,
    "unsloth-qwen2.5-coder-14b": 8,
    "phi-3.5-mini-ir": 9,             # los más rápidos
    "phi-3.5-mini": 10,
    "llama-3.2-3b": 11,
    "tinyllama-ir": 12,
}


def _force_utf8_output() -> None:
    """Pone stdout/stderr en UTF-8 para que los acentos se vean bien en Windows.

    Las consolas Windows heredadas usan cp1252, que convierte caracteres como
    'ñ'/'¿' en mojibake. Reconfigurar no hace nada donde ya se usa UTF-8.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def list_devices() -> list[str]:
    """Devuelve los dispositivos OpenVINO visibles en esta máquina."""
    import openvino as ov

    core = ov.Core()
    return list(core.available_devices)


def list_models() -> None:
    """Imprime el registro de modelos recomendados (GGUF e IR)."""
    print("Modelos recomendados (alias -> fichero/directorio):")
    for alias, info in MODELS.items():
        objetivo = info.get("file") or (info.get("dir", "") + "/  [IR]")
        print(f"  {alias:16s} {info['size']:9s} {objetivo}")
        print(f"  {'':16s} {info['note']}")


def _ajustar_hf() -> None:
    """Reduce el ruido de logs de Hugging Face Hub.

    ``HF_TOKEN`` (si está en el entorno) lo usa huggingface_hub automáticamente
    para límites de tasa más altos; aquí sólo bajamos la verbosidad del aviso.
    """
    try:
        from huggingface_hub.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:  # noqa: BLE001 - opcional, no crítico
        pass


def _descargar_ir(repo: str, target_dir: Path) -> Path:
    """Descarga un repo de modelo IR completo (snapshot) en ``target_dir``."""
    _ajustar_hf()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub no está instalado. Ejecuta:\n"
            "    python -m pip install --upgrade huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n⬇  Descargando modelo IR (repo completo)")
    print(f"   repo:    {repo}")
    print(f"   destino: {target_dir}")
    print("   (las barras de progreso las muestra Hugging Face)\n")
    inicio = time.perf_counter()
    snapshot_download(repo_id=repo, local_dir=str(target_dir))
    print(f"\n✓  Descargado: {target_dir}  ({_tam_dir(target_dir)}, "
          f"{time.perf_counter() - inicio:.0f}s)")
    return target_dir


def dir_gguf(base) -> Path:
    """Subdirectorio donde viven los GGUF (``<models-dir>/gguf``)."""
    return Path(base) / "gguf"


def dir_ir(base) -> Path:
    """Subdirectorio donde viven los modelos IR (``<models-dir>/ir``)."""
    return Path(base) / "ir"


def download_model(target: str, dest_dir: Path) -> Path:
    """Descarga un modelo y devuelve su ruta.

    Los GGUF van a ``<dest_dir>/gguf`` y los IR a ``<dest_dir>/ir/<dir>``.
    ``target`` es un alias del registro (GGUF o IR) o ``repo_id:fichero`` para un
    GGUF de cualquier repo de Hugging Face.
    """
    if target in MODELS:
        info = MODELS[target]
        if info.get("format") == "ir":
            return _descargar_ir(info["repo"], dir_ir(dest_dir) / info["dir"])
        repo, filename = info["repo"], info["file"]
    elif ":" in target:
        repo, filename = target.split(":", 1)
    else:
        print(
            f"Modelo desconocido '{target}'. Usa un alias conocido (ver "
            "--list-models) o la forma explícita 'repo_id:fichero.gguf'.",
            file=sys.stderr,
        )
        sys.exit(2)

    _ajustar_hf()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "huggingface_hub no está instalado. Ejecuta:\n"
            "    python -m pip install --upgrade huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)

    destino = dir_gguf(dest_dir)
    destino.mkdir(parents=True, exist_ok=True)
    print(f"\n⬇  Descargando: {filename}")
    print(f"   repo:    {repo}")
    print(f"   destino: {destino}")
    print("   (las barras de progreso las muestra Hugging Face)\n")
    inicio = time.perf_counter()
    path = hf_hub_download(
        repo_id=repo, filename=filename, local_dir=str(destino)
    )
    resolved = Path(path)
    print(f"\n✓  Descargado: {resolved}  "
          f"({resolved.stat().st_size / 1e9:.2f} GB, "
          f"{time.perf_counter() - inicio:.0f}s)")
    return resolved


def _params_de_nombre(nombre: str) -> float | None:
    """Extrae los billones de parámetros del nombre (p. ej. '14B' -> 14.0)."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b",
                  nombre.replace("-", " ").replace("_", " ").lower())
    return float(m.group(1)) if m else None


def _filtra_candidatos(ids: list[str], *, min_b=0.0, max_b=1e9) -> list[dict]:
    """Filtra ids de modelos de HF por tamaño (B) inferido del nombre."""
    salida = []
    for mid in ids:
        b = _params_de_nombre(mid)
        if b is not None and not (min_b <= b <= max_b):
            continue
        salida.append({"repo": mid, "params_b": b})
    return salida


def buscar_modelos(*, formato: str = "gguf", consulta: str = "",
                   min_b: float = 0.0, max_b: float = 14.0,
                   limite: int = 40) -> list[dict]:
    """Busca modelos en Hugging Face por formato/tamaño (devuelve repos candidatos).

    ``formato='ir'`` busca en la org OpenVINO (repos ``*-ov``); ``'gguf'`` busca
    repos GGUF. Filtra por tamaño (B inferido del nombre). Devuelve
    ``[{"repo", "params_b"}]``. Requiere ``huggingface_hub`` y red.
    """
    _ajustar_hf()
    from huggingface_hub import HfApi

    api = HfApi()
    if formato == "ir":
        modelos = api.list_models(author="OpenVINO",
                                  search=(consulta or "int4-ov"), limit=300)
        ids = [m.id for m in modelos if m.id.endswith("-ov")]
    else:
        texto = (consulta + " GGUF").strip()
        modelos = api.list_models(search=texto, limit=300)
        ids = [m.id for m in modelos if m.id.upper().endswith("GGUF")]
    return _filtra_candidatos(ids, min_b=min_b, max_b=max_b)[:limite]


def _ficheros_gguf(repo: str, cuant: str = "Q4_K_M") -> list[str]:
    """Lista los ficheros .gguf de ``repo`` que contienen la cuantización dada."""
    _ajustar_hf()
    from huggingface_hub import HfApi

    ficheros = HfApi().list_repo_files(repo)
    coincidentes = [f for f in ficheros
                    if f.endswith(".gguf") and cuant.lower() in f.lower()]
    return coincidentes or [f for f in ficheros if f.endswith(".gguf")]


def borrar_modelo_disco(path) -> None:
    """Borra del disco un modelo (fichero GGUF o directorio IR) y su registro perf."""
    import shutil as _sh
    ruta = Path(path)
    nombre = ruta.name
    if ruta.is_dir():
        _sh.rmtree(ruta, ignore_errors=True)
    elif ruta.is_file():
        ruta.unlink()
    try:
        perf_db.borrar(nombre)
    except Exception:  # noqa: BLE001 - best-effort
        pass


def convert_to_ir(model_id: str, dest_dir: Path, weight_format: str = "int4") -> Path:
    """Convierte un modelo de Hugging Face a OpenVINO IR con ``optimum-cli``.

    Exporta a ``dest_dir/<nombre>-<weight_format>-ov`` (un directorio IR con
    ``openvino_model.xml``/``.bin`` + tokenizador) que ``LLMPipeline`` carga con
    ``mmap``. Requiere ``optimum-intel[openvino]``.

    Args:
      model_id: Id de Hugging Face (p. ej. ``Qwen/Qwen2.5-14B-Instruct``).
      dest_dir: Directorio donde crear la carpeta IR.
      weight_format: Cuantización de pesos (``int4``/``int8``/``fp16``).

    Returns:
      La ruta del directorio IR generado.
    """
    exe = shutil.which("optimum-cli")
    if exe:
        base = [exe]
    else:
        try:
            import optimum  # noqa: F401 - solo para comprobar que está instalado
        except ImportError:
            print(
                "optimum-intel no está instalado. Ejecuta:\n"
                '    python -m pip install "optimum-intel[openvino]"',
                file=sys.stderr,
            )
            sys.exit(1)
        base = [sys.executable, "-m", "optimum.commands.optimum_cli"]

    nombre = model_id.rstrip("/").split("/")[-1]
    salida = Path(dest_dir) / f"{nombre}-{weight_format}-ov"
    salida.parent.mkdir(parents=True, exist_ok=True)
    cmd = base + ["export", "openvino", "--model", model_id,
                  "--weight-format", weight_format, str(salida)]
    print(f"Convirtiendo {model_id} a IR ({weight_format}) en {salida} ...")
    print(f"  $ {' '.join(cmd)}")
    returncode = subprocess.call(cmd)  # noqa: S603 - array de args, sin shell
    if returncode != 0:
        print(f"La conversión a IR falló (código {returncode}).", file=sys.stderr)
        sys.exit(returncode)
    print(f"IR guardado en: {salida}")
    return salida


def describe_devices() -> None:
    """Imprime los dispositivos OpenVINO disponibles con su nombre de producto."""
    import openvino as ov

    core = ov.Core()
    devices = core.available_devices
    if not devices:
        print("No se enumeró ningún dispositivo OpenVINO.")
        return
    print("Dispositivos OpenVINO disponibles:")
    for name in devices:
        try:
            full_name = core.get_property(name, "FULL_DEVICE_NAME")
        except Exception:  # noqa: BLE001 - la propiedad es opcional por dispositivo
            full_name = "<desconocido>"
        print(f"  {name:8s} -> {full_name}")


def pick_device(requested: str) -> str:
    """Resuelve el dispositivo en el que ejecutar.

    "AUTO" prefiere una GPU Intel (la Iris Xe se enumera como ``GPU``) y recurre
    a la CPU. Un nombre de dispositivo explícito se valida contra los presentes.
    """
    devices = list_devices()
    if requested.upper() == "AUTO":
        gpu = next((d for d in devices if d.startswith("GPU")), None)
        chosen = gpu or "CPU"
        print(f"Dispositivo autoseleccionado: {chosen}")
        return chosen

    if requested not in devices:
        print(
            f"Dispositivo solicitado '{requested}' no encontrado. "
            f"Disponibles: {', '.join(devices) or '(ninguno)'}",
            file=sys.stderr,
        )
        sys.exit(2)
    return requested


def es_modelo_ir(model_path) -> bool:
    """Indica si ``model_path`` es un directorio de modelo OpenVINO IR.

    Un modelo IR es una carpeta con ``openvino_model.xml`` (y su ``.bin``), tal
    como lo produce ``optimum-cli export openvino``. ``LLMPipeline`` lo carga con
    ``mmap`` (los pesos se paginan desde el ``.bin``).
    """
    ruta = Path(model_path)
    return ruta.is_dir() and (ruta / "openvino_model.xml").is_file()


# Directorio de caché de compilación de OpenVINO (acelera recargas posteriores).
OV_CACHE_DIR = Path("models") / ".ovcache"


def ov_plugin_config(device: str) -> dict:
    """Config de plugin OpenVINO según el dispositivo.

    - ``CACHE_DIR``: cachea el modelo compilado en disco, de modo que las cargas
      siguientes del mismo modelo/dispositivo sean mucho más rápidas.
    - En GPU, ``KV_CACHE_PRECISION=f16`` desactiva la compresión de la caché KV
      (a u8 por defecto), que rompe la API ``get_state`` de la pipeline de chat
      ("get_state API is supported only when KV-cache compression is disabled").
    """
    config: dict = {}
    try:
        OV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        config["CACHE_DIR"] = str(OV_CACHE_DIR)
    except OSError:
        pass
    if device.upper().startswith("GPU"):
        try:
            import openvino as ov
            config["KV_CACHE_PRECISION"] = ov.Type.f16
        except Exception:  # noqa: BLE001 - como respaldo, la cadena
            config["KV_CACHE_PRECISION"] = "f16"
    return config


def build_pipeline(model_path: Path, device: str):
    """Construye una LLMPipeline de OpenVINO GenAI.

    ``model_path`` puede ser un ``.gguf`` o un **directorio IR** (``.xml``/``.bin``,
    cargado con ``mmap``). En GPU se desactiva la compresión de la caché KV
    (ver :func:`ov_plugin_config`). Un fallo de carga en GPU reintenta en CPU.
    """
    try:
        import openvino_genai as ov_genai
    except ImportError:
        print(
            "openvino-genai no está instalado. Ejecuta:\n"
            "    python -m pip install --upgrade openvino openvino-genai",
            file=sys.stderr,
        )
        sys.exit(1)

    tipo = "IR (mmap)" if es_modelo_ir(model_path) else "GGUF"
    print(f"Cargando modelo {tipo} '{Path(model_path).name}' en {device} ...")
    start = time.perf_counter()
    try:
        pipe = ov_genai.LLMPipeline(str(model_path), device,
                                    **ov_plugin_config(device))
    except Exception as exc:  # noqa: BLE001 - mostrar el error real de OpenVINO
        print(f"Falló la carga del modelo en {device}: {exc}", file=sys.stderr)
        if device.startswith("GPU"):
            print("Reintentando en CPU ...", file=sys.stderr)
            pipe = ov_genai.LLMPipeline(str(model_path), "CPU")
        else:
            raise
    elapsed = time.perf_counter() - start
    print(f"Modelo listo en {elapsed:.1f}s")
    try:
        perf_db.registrar_carga(
            Path(model_path).name, elapsed,
            formato="IR" if es_modelo_ir(model_path) else "GGUF", motor="openvino")
    except Exception:  # noqa: BLE001 - la BD de rendimiento es best-effort
        pass
    return pipe


def _hay_binario_llamacpp() -> bool:
    """Indica si hay un binario de llama.cpp disponible (para el reparto)."""
    try:
        llama_engine.find_binary()
        return True
    except llama_engine.LlamaCppError:
        return False


# Caché de detección de motor para GGUF locales: (ruta, mtime) -> motor.
_CACHE_MOTOR_LOCAL: dict[tuple, str] = {}


def _detectar_motor_gguf(model_path) -> str:
    """Detecta el motor para un GGUF leyendo su cabecera (rápida y cacheada).

    Devuelve ``openvino`` si la arquitectura es nativa de OpenVINO y no es una
    familia problemática; ``llamacpp`` si la arquitectura no la soporta OpenVINO
    o es Mistral/Mixtral (que OpenVINO GenAI no lee); ``auto`` si no se puede leer.
    """
    ruta = Path(model_path)
    try:
        clave = (str(ruta), ruta.stat().st_mtime_ns)
    except OSError:
        return "auto"
    if clave in _CACHE_MOTOR_LOCAL:
        return _CACHE_MOTOR_LOCAL[clave]
    try:
        meta = gguf_reader.read_scalar_metadata(ruta)["metadata"]
    except (OSError, ValueError):
        _CACHE_MOTOR_LOCAL[clave] = "auto"
        return "auto"
    arch = str(meta.get("general.architecture") or "").lower()
    nombre = " ".join(
        str(meta.get(k, "")) for k in
        ("general.name", "general.basename", "general.finetune")).lower()
    if any(fam in arch or fam in nombre for fam in ("mistral", "mixtral")):
        motor = "llamacpp"          # OpenVINO GenAI no lee bien estos GGUF
    elif arch in OPENVINO_ARCHS:
        motor = "openvino"
    elif arch:
        motor = "llamacpp"          # arquitectura no soportada por OpenVINO
    else:
        motor = "auto"
    _CACHE_MOTOR_LOCAL[clave] = motor
    return motor


def _engine_hint(model_path) -> str:
    """Pista de motor para un modelo: ``openvino``, ``llamacpp`` o ``auto``.

    Un directorio IR es siempre OpenVINO. Para GGUF: primero el registro MODELS
    (por nombre de fichero); si no está, se detecta leyendo la cabecera del GGUF.
    """
    if es_modelo_ir(model_path):
        return "openvino"
    nombre = Path(model_path).name
    for info in MODELS.values():
        if info.get("file") == nombre:
            return info.get("engine", "auto")
    return _detectar_motor_gguf(model_path)


def etiqueta_motor(engine_hint: str) -> str:
    """Texto con marcado rich para mostrar la pista de motor en el selector."""
    return {
        "openvino": "[green]OpenVINO[/]",
        "llamacpp": "[yellow]llama.cpp (respaldo)[/]",
    }.get(engine_hint, "[dim]auto[/]")


def crear_motor(model_path, args, device: str, *, verbose: bool = True) -> dict:
    """Elige y construye el motor de inferencia para ``model_path``.

    Según ``args.engine`` (auto/openvino/llamacpp) y el reparto de capas que
    calcule :mod:`layer_planner`, devuelve un descriptor de motor:
    ``{"kind", "obj", "plan", "n_gpu_layers", "device"}``.

    - ``openvino``: carga el GGUF entero en el dispositivo (sin mmap).
    - ``llamacpp``: abre el GGUF con mmap y descarga ``-ngl`` capas a la GPU,
      dejando el resto en la RAM de la CPU.
    - ``auto``: usa llama.cpp con reparto cuando el modelo NO cabe entero en la
      VRAM y hay binario disponible; si no, OpenVINO.
    """
    import openvino_genai as ov_genai

    sistema = construir_sistema(model_path)  # identidad LlamaVINO + primer de ficheros

    # Directorio IR: siempre OpenVINO (carga con mmap del .bin); sin planner ni
    # respaldo a llama.cpp (llama.cpp no lee IR).
    if es_modelo_ir(model_path):
        pipe = build_pipeline(Path(model_path), device)
        return {"kind": "openvino", "obj": pipe, "plan": None, "sistema": sistema,
                "n_gpu_layers": None, "device": f"{device} (IR, mmap)"}

    pref = getattr(args, "engine", "auto")
    n_ctx = int(getattr(args, "n_ctx", 4096))
    ngl_pref = getattr(args, "n_gpu_layers", "auto")

    # Plan de reparto en tiempo real (lee la cabecera + mide VRAM/RAM).
    plan = None
    try:
        plan = layer_planner.plan_offload(str(model_path), device=device, n_ctx=n_ctx)
    except (OSError, ValueError) as exc:
        if verbose:
            print(f"No se pudo planificar el reparto de capas: {exc}", file=sys.stderr)

    def _ngl() -> int:
        if str(ngl_pref).lower() == "auto":
            return plan.n_gpu_layers if plan is not None else 0
        return int(ngl_pref)

    if pref == "llamaserver":
        ngl = _ngl()
        if verbose and plan is not None:
            print(plan.reason)
        _t0 = time.perf_counter()
        engine = llama_engine.LlamaServerEngine(
            str(model_path), n_gpu_layers=ngl, n_ctx=n_ctx, use_mmap=True,
            n_probs=int(getattr(args, "n_probs", 5)))
        try:
            perf_db.registrar_carga(Path(model_path).name,
                                    time.perf_counter() - _t0,
                                    formato="GGUF", motor="llamaserver")
        except Exception:  # noqa: BLE001 - best-effort
            pass
        return {"kind": "llamaserver", "obj": engine, "plan": plan, "sistema": sistema,
                "n_gpu_layers": ngl,
                "device": f"llama-server (ngl={ngl}, mmap, candidatas)"}

    usar_llama = pref == "llamacpp"
    if pref == "auto":
        # Respaldo a llama.cpp si el modelo no lo soporta OpenVINO (pista del
        # registro) o no cabe entero en VRAM, y hay binario disponible.
        no_cabe = plan is not None and not plan.fits_full_gpu
        if (_engine_hint(model_path) == "llamacpp" or no_cabe) \
                and _hay_binario_llamacpp():
            usar_llama = True

    def _motor_llamacpp() -> dict:
        ngl = _ngl()
        if verbose and plan is not None:
            print(plan.reason)
        engine = llama_engine.LlamaCppEngine(
            str(model_path), n_gpu_layers=ngl, n_ctx=n_ctx, use_mmap=True)
        return {"kind": "llamacpp", "obj": engine, "plan": plan, "sistema": sistema,
                "n_gpu_layers": ngl, "device": f"llama.cpp (ngl={ngl}, mmap)"}

    if usar_llama:
        return _motor_llamacpp()

    # OpenVINO: si no puede leer el GGUF (arquitectura/metadatos no soportados,
    # p. ej. "invalid map<K,T> key"), recae automáticamente en llama.cpp.
    try:
        pipe = build_pipeline(Path(model_path), device)
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de carga de OpenVINO
        if pref == "openvino":
            raise  # el usuario forzó OpenVINO: no enmascaramos el error
        if not _hay_binario_llamacpp():
            raise llama_engine.LlamaCppError(
                f"OpenVINO GenAI no pudo leer este GGUF ({exc}). El modelo no es "
                "compatible con OpenVINO; instala el binario de llama.cpp (define "
                "LLAMA_CPP_BIN o colócalo en ./vendor/llama.cpp/) para usar el "
                "motor de respaldo con --engine llamacpp, o elige un modelo de "
                "arquitectura llama (p. ej. llama-3.2-3b)."
            ) from exc
        if verbose:
            print(f"OpenVINO no pudo cargar el modelo ({exc}); recurriendo a "
                  "llama.cpp (mmap).", file=sys.stderr)
        return _motor_llamacpp()

    return {"kind": "openvino", "obj": pipe, "plan": plan, "sistema": sistema,
            "n_gpu_layers": None, "device": device}


def cerrar_motor(engine: dict) -> None:
    """Cierra el motor si tiene recursos que liberar (p. ej. llama-server)."""
    obj = engine.get("obj") if isinstance(engine, dict) else None
    cerrar = getattr(obj, "close", None)
    if callable(cerrar):
        cerrar()


def _motor_generar(engine: dict, turns: list[dict], settings: dict, ov_genai,
                   streamer, on_candidates=None, debe_parar=None) -> tuple[str, int]:
    """Genera una respuesta con el motor activo. Devuelve ``(texto, n_tokens)``.

    Unifica los motores: OpenVINO GenAI (``pipe.generate``), llama.cpp (CLI con
    reparto de capas) y llama-server (igual + ``on_candidates`` con las palabras
    candidatas por token). El ``streamer`` recibe cada fragmento de texto.
    ``debe_parar()`` (opcional) permite cancelar: en OpenVINO el streamer devuelve
    True para detener; en llama.cpp/server se usa ``engine.obj.cancelar()``.
    """
    contador = {"n": 0}
    # Identidad LlamaVINO (+ primer de ficheros) como mensaje de sistema, en todos
    # los motores. Si falta (compat.), recae en el primer de ficheros.
    sistema = engine.get("sistema") or FILE_PRIMER

    if engine["kind"] in ("llamacpp", "llamaserver"):
        def _stream(piece: str) -> None:
            contador["n"] += 1
            streamer(piece)

        extra = {}
        if engine["kind"] == "llamaserver":
            extra["on_candidates"] = on_candidates
        turns_sys = [{"role": "system", "content": sistema}] + turns
        texto = engine["obj"].generate(
            turns_sys, max_new_tokens=settings["max_new_tokens"],
            temperature=settings["temperature"], top_p=settings["top_p"],
            top_k=settings["top_k"], streamer=_stream, **extra)
        return texto, contador["n"]

    # OpenVINO GenAI: se reconstruye el ChatHistory en cada turno.
    config = config_from_settings(settings, ov_genai)
    history = ov_genai.ChatHistory()
    history.append({"role": "system", "content": sistema})
    for turn in turns:
        history.append(turn)

    def _stream_ov(subword: str) -> bool:
        contador["n"] += 1
        streamer(subword)
        return bool(debe_parar and debe_parar())  # True detiene la generación

    result = engine["obj"].generate(history, config, _stream_ov)
    return str(result), contador["n"]


# Parámetros comunes de muestreo de un LLM, con su tipo, rango válido y una breve
# explicación en español. Lo usan tanto el CLI como el editor interactivo
# ``/configuration``, de modo que haya una única fuente de verdad.
GEN_PARAM_SPECS: dict[str, dict] = {
    "max_new_tokens": {
        "type": int, "min": 1, "max": 32768,
        "label": "Máximo de tokens",
        "help": "Longitud máxima de la respuesta (1 token ≈ 4 caracteres).",
    },
    "temperature": {
        "type": float, "min": 0.0, "max": 2.0,
        "label": "Temperatura",
        "help": "Aleatoriedad. 0–0.3 preciso, 0.7 equilibrado, 1.0+ creativo. "
                "0 = determinista (greedy).",
    },
    "top_p": {
        "type": float, "min": 0.0, "max": 1.0,
        "label": "Top-P (nucleus)",
        "help": "Probabilidad acumulada de las palabras candidatas (0.9 típico). "
                "Mueve esto o la temperatura, no ambos.",
    },
    "top_k": {
        "type": int, "min": 0, "max": 1000,
        "label": "Top-K",
        "help": "Número fijo de palabras candidatas (p. ej. 40). 0 = sin límite.",
    },
    "presence_penalty": {
        "type": float, "min": -2.0, "max": 2.0,
        "label": "Penalización de presencia",
        "help": "Castiga palabras por haber aparecido (fomenta temas nuevos). "
                "-2.0 a 2.0; positivo bajo (0.1–0.5) reduce redundancia.",
    },
    "frequency_penalty": {
        "type": float, "min": -2.0, "max": 2.0,
        "label": "Penalización de frecuencia",
        "help": "Castiga palabras según cuánto se repiten. -2.0 a 2.0; "
                "positivo bajo evita repetir las mismas frases.",
    },
    "stop_strings": {
        "type": list, "label": "Secuencias de parada",
        "help": "Cadenas que detienen la generación al aparecer "
                "(p. ej. 'Usuario:'). Separa varias con comas.",
    },
}


def default_gen_settings(args=None) -> dict:
    """Devuelve un dict de ajustes mutable, sembrado por defecto (o con ``args``)."""
    settings = {
        "max_new_tokens": 512,
        "temperature": 0.0,  # por defecto determinista (greedy)
        "top_p": 0.9,
        "top_k": 40,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "stop_strings": [],
    }
    if args is not None:
        settings.update(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            presence_penalty=getattr(args, "presence_penalty", 0.0),
            frequency_penalty=getattr(args, "frequency_penalty", 0.0),
            stop_strings=list(getattr(args, "stop", None) or []),
        )
    return settings


# Fichero donde se persisten los ajustes interactivos entre sesiones.
AJUSTES_PATH = Path(".llamavino.json")


def cargar_ajustes() -> dict:
    """Carga los ajustes persistidos (generación + color), o ``{}`` si no hay."""
    try:
        datos = json.loads(AJUSTES_PATH.read_text(encoding="utf-8"))
        return datos if isinstance(datos, dict) else {}
    except (OSError, ValueError):
        return {}


def guardar_ajustes(settings: dict | None = None, color: str | None = None,
                    **extra) -> None:
    """Persiste ajustes haciendo *merge* con lo ya guardado (no pisa otras claves).

    Acepta ``settings``/``color`` y cualquier clave extra (p. ej. ``last_model``).
    """
    datos = cargar_ajustes()
    if settings is not None:
        datos["settings"] = settings
    if color is not None:
        datos["color"] = color
    datos.update(extra)
    try:
        AJUSTES_PATH.write_text(
            json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def _generation_config(
    ov_genai,
    *,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    stop_strings=None,
):
    """Construye un GenerationConfig de OpenVINO GenAI con valores explícitos."""
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens
    if temperature > 0:
        config.do_sample = True
        config.temperature = temperature
        config.top_p = top_p
        config.top_k = top_k
    else:
        config.do_sample = False
    # Los controles de repetición se aplican haya o no muestreo.
    config.presence_penalty = presence_penalty
    config.frequency_penalty = frequency_penalty
    if stop_strings:
        config.stop_strings = set(stop_strings)
        config.include_stop_str_in_output = False
    return config


def config_from_settings(settings: dict, ov_genai):
    """Construye un GenerationConfig desde un dict estilo ``default_gen_settings``."""
    return _generation_config(
        ov_genai,
        max_new_tokens=settings["max_new_tokens"],
        temperature=settings["temperature"],
        top_p=settings["top_p"],
        top_k=settings["top_k"],
        presence_penalty=settings.get("presence_penalty", 0.0),
        frequency_penalty=settings.get("frequency_penalty", 0.0),
        stop_strings=settings.get("stop_strings") or None,
    )


def make_generation_config(args, ov_genai):
    """Traduce los argumentos del CLI a un GenerationConfig de OpenVINO GenAI."""
    return config_from_settings(default_gen_settings(args), ov_genai)


# --------------------------------------------------------------------------- #
# Escritura en disco de los archivos generados por el modelo
# --------------------------------------------------------------------------- #

# Primer de sistema que enseña al modelo a marcar el archivo que quiere escribir.
# Se antepone al historial para que hasta un modelo pequeño emita una pista legible.
FILE_PRIMER = (
    "Puedes crear archivos en el espacio de trabajo del usuario. Cuando generes "
    "el contenido completo de un archivo, escríbelo en un bloque de código "
    "cercado e indica el nombre del archivo en la línea de apertura, por "
    "ejemplo:\n```python hola.py\nprint('hola')\n```\n"
    "Usa rutas relativas dentro del proyecto."
)


def descripcion_modelo(model_path) -> str:
    """Nombre/arquitectura reales del modelo cargado (para «¿en qué te basas?»)."""
    if not model_path:
        return "desconocido"
    try:
        if es_modelo_ir(model_path):
            datos = json.loads(
                (Path(model_path) / "config.json").read_text(encoding="utf-8"))
            nombre = datos.get("_name_or_path") or Path(model_path).name
            arch = datos.get("model_type") or (
                (datos.get("architectures") or [None])[0])
        else:
            meta = gguf_reader.read_scalar_metadata(model_path)["metadata"]
            nombre = meta.get("general.name") or Path(model_path).name
            arch = meta.get("general.architecture")
        return f"{nombre}" + (f" (arquitectura {arch})" if arch else "")
    except Exception:  # noqa: BLE001 - si no se puede leer, el nombre del fichero
        return Path(model_path).name


def construir_sistema(model_path) -> str:
    """Prompt de sistema con la identidad LlamaVINO + el primer de escritura.

    La identidad es fija (te llamas LlamaVINO, no el modelo base), pero ante la
    pregunta del modelo fundacional se responde con el dato real del GGUF/IR.
    """
    return (
        "Eres LlamaVINO, un asistente que se ejecuta en local combinando la "
        "lectura de modelos GGUF de llama.cpp con la librería OpenVINO para GPUs "
        "Intel. Tu nombre es LlamaVINO: preséntate siempre así y NO digas que "
        "eres Claude ni te identifiques con el nombre del modelo base. Sólo si te "
        "preguntan explícitamente en qué modelo fundacional estás basado, responde "
        f"con el dato real y exacto: {descripcion_modelo(model_path)}.\n\n"
        + FILE_PRIMER
    )

# Un token con pinta de nombre de archivo: nombre + extensión (1–8 alfanuméricos).
_FILENAME_RE = re.compile(r"[\w./\\-]+\.[A-Za-z0-9]{1,8}")
# Bloque de código cercado: captura la línea de info y el cuerpo.
_CODE_BLOCK_RE = re.compile(r"```([^\n]*)\n(.*?)```", re.DOTALL)
# Marcadores de comentario iniciales para detectar un nombre en la 1ª línea.
_COMMENT_PREFIX_RE = re.compile(r"^\s*(?:#|//|--|<!--|/\*|;)\s*")


def _filename_from_info(info: str) -> str | None:
    """Extrae un nombre de archivo de la línea de info del bloque, si lo hay.

    Reconoce ``python foo.py``, ``path=foo.py``, ``file="foo.py"`` y similares.
    La palabra del lenguaje a secas (sin punto) se ignora.
    """
    info = info.strip()
    if not info:
        return None
    # Formas clave=valor: path=, file=, filename=, title=, name=.
    kv = re.search(
        r'(?:path|file|filename|title|name)\s*=\s*["\']?([^\s"\']+)', info,
        re.IGNORECASE,
    )
    if kv:
        return kv.group(1)
    for token in re.split(r"[\s:]+", info):
        if _FILENAME_RE.fullmatch(token):
            return token
    return None


def _filename_from_first_line(code: str) -> str | None:
    """Extrae un nombre de archivo de un comentario inicial como ``# foo.py``."""
    first = code.splitlines()[0] if code.strip() else ""
    stripped = _COMMENT_PREFIX_RE.sub("", first).strip().rstrip("-->*/").strip()
    # Permite una etiqueta opcional "file:"/"archivo:" antes del nombre.
    stripped = re.sub(r"^(?:file|archivo|fichero)\s*:\s*", "", stripped,
                      flags=re.IGNORECASE)
    if _FILENAME_RE.fullmatch(stripped):
        return stripped
    return None


def _filename_from_preceding(text: str) -> str | None:
    """Busca un nombre de archivo justo antes del bloque (p. ej. ```foo.py```)."""
    tail = text[-200:]
    candidates = _FILENAME_RE.findall(tail)
    return candidates[-1] if candidates else None


def extract_code_blocks(text: str) -> list[dict]:
    """Devuelve los bloques de código cercados con su mejor nombre de archivo.

    Cada elemento: ``{"lang", "filename" (str|None), "code"}``. El nombre se
    infiere de la línea de info, un comentario inicial o el texto anterior.
    """
    blocks: list[dict] = []
    for match in _CODE_BLOCK_RE.finditer(text):
        info, code = match.group(1), match.group(2)
        lang = info.strip().split()[0] if info.strip() else ""
        filename = (
            _filename_from_info(info)
            or _filename_from_first_line(code)
            or _filename_from_preceding(text[: match.start()])
        )
        blocks.append({"lang": lang, "filename": filename, "code": code})
    return blocks


def write_workspace_file(filename: str, code: str, workspace: Path) -> Path:
    """Escribe ``code`` en ``filename``, confinado dentro de ``workspace``.

    Devuelve la ruta resuelta. Lanza ``edit_formats.PatchError`` si la ruta se
    sale del espacio de trabajo.
    """
    target = _confined_path(filename, workspace)
    target.parent.mkdir(parents=True, exist_ok=True)
    edit_formats.write_text_with_retry(target, code)
    return target


# --------------------------------------------------------------------------- #
# Backend de líneas JSON para el frontend Ink (Node)
# --------------------------------------------------------------------------- #


def _models_payload(models_dir: str) -> list[dict]:
    """Registro + modelos locales como filas serializables a JSON para la UI."""
    out = []
    for row in _build_model_menu(Path(models_dir)):
        out.append(
            {
                "name": row["name"],
                "size": row["size"],
                "note": row["note"],
                "downloaded": row["downloaded"],
                "path": str(row["path"]) if row["path"] else None,
                "target": row["target"],
                "engine": row.get("engine", "auto"),
                "format": row.get("format", "GGUF"),
                "publisher": row.get("publisher", ""),
                "highlight": row.get("highlight", False),
                "recomendacion": row.get("recomendacion"),
                "tok_s": row.get("tok_s"),
            }
        )
    return out


JSONLD_CONTEXT = "https://llamavino.dev/ns/v1"


def _ld(type_: str, **fields) -> dict:
    """Envuelve una carga útil del protocolo como documento JSON-LD.

    Cada mensaje es un pequeño documento de datos enlazados: ``@context`` nombra
    el vocabulario, ``@type`` el tipo de mensaje y ``@id`` (cuando existe) liga
    una respuesta con la petición que la originó.
    """
    return {"@context": JSONLD_CONTEXT, "@type": type_, **fields}


def _init_telemetry():
    """Configura OpenTelemetry con un exportador de spans a consola, si está el SDK.

    Sigue las convenciones semánticas GenAI. Devuelve un tracer, o None cuando el
    SDK de OTel no está instalado, de modo que la telemetría es estrictamente
    opcional y nunca bloquea en una máquina con pocos recursos. Usa un exportador
    a consola en stderr para no necesitar un colector externo; las variables
    OTEL_* siguen aplicándose.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
    except ImportError:
        return None

    provider = TracerProvider(resource=Resource.create({"service.name": "llamavino"}))
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter(out=sys.stderr)))
    trace.set_tracer_provider(provider)
    return trace.get_tracer("llamavino")


def _confined_path(path_str: str, root: Path) -> Path:
    """Resuelve una ruta y garantiza que queda dentro de la raíz ``root``.

    Implementa la regla de endurecimiento de directorios del harness del agente:
    las rutas relativas se toman desde ``root`` y cualquier ruta que se escape de
    ``root`` (p. ej. con ``..``) se rechaza.

    Args:
      path_str: La ruta solicitada (absoluta o relativa).
      root: La raíz del espacio de trabajo en la que debe permanecer la ruta.

    Returns:
      La ruta absoluta resuelta y confinada.

    Raises:
      edit_formats.PatchError: Si la ruta se sale del espacio de trabajo.
    """
    root = root.resolve()
    target = Path(path_str)
    if not target.is_absolute():
        target = root / target
    target = target.resolve()
    if target != root and root not in target.parents:
        raise edit_formats.PatchError(f"ruta fuera del workspace: {path_str}")
    return target


def _load_mcp_config(workspace: Path) -> dict:
    """Carga servidores MCP pre-aprobados de ``mcp_config.json`` del workspace.

    Por seguridad, los comandos de los servidores MCP vienen sólo de este fichero
    en disco (que controla el usuario), nunca del modelo ni del protocolo, que se
    refieren a un servidor por su nombre. Devuelve un mapa ``nombre -> entrada``
    (vacío si no hay configuración).
    """
    path = workspace / "mcp_config.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data.get("servers", {})


def _connect_mcp(name: str, config: dict, workspace: Path) -> "mcp_client.MCPClient":
    """Lanza e inicializa un servidor MCP pre-aprobado por su nombre.

    Raises:
      mcp_client.MCPError: Si el nombre del servidor no está en la configuración.
    """
    entry = config.get(name)
    if entry is None:
        raise mcp_client.MCPError(f"servidor MCP no configurado: {name}")
    command = list(entry["command"])
    if command and command[0] in ("python", "python3"):
        command[0] = sys.executable  # usa el mismo intérprete
    client = mcp_client.MCPClient(command, cwd=str(workspace))
    client.initialize()
    return client


def serve_stdio() -> int:
    """Ejecuta un protocolo de líneas JSON-LD en stdin/stdout para el frontend Ink.

    Cada línea es un documento JSON-LD. Las peticiones usan valores ``@type`` como
    ``Generate``; las respuestas usan ``Token``, ``Done``, ``Error`` y similares.
    La pipeline cargada se mantiene en memoria entre peticiones. La salida del
    protocolo va al stdout real; cualquier otra impresión se redirige a stderr
    para que nunca corrompa el flujo.

    Returns:
      Código de salida del proceso (0 en un cierre limpio).
    """
    _force_utf8_output()
    real_out = sys.stdout
    sys.stdout = sys.stderr  # impresiones incidentales (pick_device, descargas) -> stderr

    def send(type_: str, **fields) -> None:
        real_out.write(json.dumps(_ld(type_, **fields), ensure_ascii=False) + "\n")
        real_out.flush()

    try:
        import openvino_genai as ov_genai
    except ImportError as exc:
        send("Fatal", message=f"openvino-genai no disponible: {exc}")
        return 1

    state: dict = {"pipe": None, "model": None, "device": None,
                   "engine": "openvino", "llama": None}
    tracer = _init_telemetry()
    workspace = Path.cwd().resolve()  # raíz para el confinamiento de ficheros
    mcp_config = _load_mcp_config(workspace)
    mcp_clients: dict[str, mcp_client.MCPClient] = {}

    def get_mcp(name: str) -> "mcp_client.MCPClient":
        if name not in mcp_clients:
            mcp_clients[name] = _connect_mcp(name, mcp_config, workspace)
        return mcp_clients[name]

    send("Ready")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            send("Error", message="JSON-LD no válido")
            continue

        req_type = msg.get("@type")
        rid = msg.get("@id")
        try:
            if req_type == "ListModels":
                send("Models", **{"@id": rid,
                                  "items": _models_payload(msg.get("models_dir", "."))})
            elif req_type == "ListDevices":
                send("Devices", **{"@id": rid, "items": list_devices()})
            elif req_type == "Download":
                path = download_model(msg["target"], Path(msg.get("models_dir", ".")))
                send("Downloaded", **{"@id": rid, "path": str(path)})
            elif req_type == "Load":
                engine = msg.get("engine", "openvino")
                n_ctx = int(msg.get("n_ctx", 4096))
                started = time.perf_counter()
                dev_req = msg.get("device", "AUTO")
                plan = None
                # Motor "auto": respaldo a llama.cpp si OpenVINO no lo soporta
                # (pista del registro) o no cabe entero en VRAM.
                if engine == "auto":
                    try:
                        plan = layer_planner.plan_offload(
                            msg["model"], device=dev_req, n_ctx=n_ctx)
                    except (OSError, ValueError):
                        plan = None
                    no_cabe = plan is not None and not plan.fits_full_gpu
                    pista_llama = _engine_hint(msg["model"]) == "llamacpp"
                    if (no_cabe or pista_llama) and _hay_binario_llamacpp():
                        engine = "llamacpp"
                    else:
                        engine = "openvino"
                if engine == "llamacpp":
                    ngl = msg.get("n_gpu_layers", "auto")
                    if str(ngl).lower() == "auto":
                        if plan is None:
                            plan = layer_planner.plan_offload(
                                msg["model"], device=dev_req, n_ctx=n_ctx)
                        ngl = plan.n_gpu_layers
                    ngl = int(ngl)
                    state["llama"] = llama_engine.LlamaCppEngine(
                        msg["model"], n_gpu_layers=ngl, n_ctx=n_ctx,
                        use_mmap=bool(msg.get("use_mmap", True)),
                    )
                    state["engine"] = "llamacpp"
                    state["pipe"] = None
                    device = f"llama.cpp (ngl={ngl}, mmap)"
                else:
                    device = pick_device(dev_req)
                    state["pipe"] = ov_genai.LLMPipeline(
                        msg["model"], device, **ov_plugin_config(device))
                    state["engine"] = "openvino"
                    state["llama"] = None
                state["model"] = msg["model"]
                state["device"] = device
                payload = {"@id": rid, "device": device, "engine": engine,
                           "ms": int((time.perf_counter() - started) * 1000)}
                if plan is not None:
                    payload["plan"] = {
                        "block_count": plan.block_count,
                        "n_gpu_layers": plan.n_gpu_layers,
                        "fits_full_gpu": plan.fits_full_gpu,
                        "reason": plan.reason,
                    }
                send("Loaded", **payload)
            elif req_type == "Generate":
                temperature = float(msg.get("temperature", 0.7))
                max_new_tokens = int(msg.get("max_new_tokens", 512))
                top_p = float(msg.get("top_p", 0.9))
                top_k = int(msg.get("top_k", 40))
                presence_penalty = float(msg.get("presence_penalty", 0.0))
                frequency_penalty = float(msg.get("frequency_penalty", 0.0))
                stop_strings = msg.get("stop_strings") or None
                turns = msg.get("history", [])

                if state["engine"] == "llamacpp":
                    if state["llama"] is None:
                        send("Error", **{"@id": rid, "message": "no hay modelo cargado"})
                        continue
                    llama_tokens = 0

                    def llama_streamer(piece: str) -> None:
                        nonlocal llama_tokens
                        llama_tokens += 1
                        send("Token", **{"@id": rid, "text": piece})

                    started = time.perf_counter()
                    text = state["llama"].generate(
                        turns, max_new_tokens=max_new_tokens, temperature=temperature,
                        top_p=top_p, top_k=top_k, streamer=llama_streamer)
                    send("Done", **{"@id": rid, "text": text,
                                    "ms": int((time.perf_counter() - started) * 1000),
                                    "usage": {"output_tokens": llama_tokens}})
                    continue

                if state["pipe"] is None:
                    send("Error", **{"@id": rid, "message": "no hay modelo cargado"})
                    continue
                config = _generation_config(
                    ov_genai,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    stop_strings=stop_strings,
                )
                history = ov_genai.ChatHistory()
                input_chars = 0
                for turn in turns:
                    history.append({"role": turn["role"], "content": turn["content"]})
                    input_chars += len(turn.get("content", ""))

                output_tokens = 0

                def streamer(subword: str) -> bool:
                    nonlocal output_tokens
                    output_tokens += 1
                    send("Token", **{"@id": rid, "text": subword})
                    return False

                # Span de convención semántica GenAI; no-op si no hay OTel.
                span_cm = (
                    tracer.start_as_current_span("chat")
                    if tracer is not None
                    else contextlib.nullcontext()
                )
                started = time.perf_counter()
                with span_cm as span:
                    if span is not None:
                        span.set_attribute("gen_ai.operation.name", "chat")
                        span.set_attribute("gen_ai.system", "openvino")
                        span.set_attribute("gen_ai.request.model", state["model"] or "")
                        span.set_attribute("gen_ai.request.max_tokens", max_new_tokens)
                        span.set_attribute("gen_ai.request.temperature", temperature)
                    result = state["pipe"].generate(history, config, streamer)
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    if span is not None:
                        # El recuento de tokens de entrada no es barato; se aproxima.
                        span.set_attribute("gen_ai.usage.input_tokens", input_chars // 4)
                        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
                        span.set_attribute("gen_ai.response.id", str(rid))
                send("Done", **{"@id": rid, "text": str(result), "ms": elapsed_ms,
                                "usage": {"output_tokens": output_tokens,
                                          "input_tokens_estimated": input_chars // 4}})
            elif req_type == "FileActionRequest":
                action = msg.get("action")
                if action != "patch":
                    send("Error", **{"@id": rid,
                                     "message": f"acción no soportada: {action}"})
                    continue
                patch_type = str(msg.get("patchType", "GitUnifiedDiff")).lower()
                fmt = "aider" if ("aider" in patch_type or "search" in patch_type) else "unified"
                target = _confined_path(msg["path"], workspace)
                new_text = edit_formats.apply_patch_to_file(
                    target, msg["patch"], fmt=fmt
                )
                send("Patched", **{"@id": rid, "path": str(target),
                                   "format": fmt, "bytes": len(new_text)})
            elif req_type == "WriteFile":
                target = write_workspace_file(msg["path"], msg.get("content", ""),
                                              workspace)
                send("FileWritten", **{"@id": rid, "path": str(target),
                                       "bytes": len(msg.get("content", ""))})
            elif req_type == "GgufHeader":
                path = msg.get("path") or state.get("model")
                if not path:
                    send("Error", **{"@id": rid, "message": "no hay modelo cargado"})
                else:
                    desc = gguf_reader.describe_gguf(
                        str(path), key_filter=msg.get("filter") or None,
                        array_limit=msg.get("limit") or None)
                    send("GgufInfo", **{"@id": rid, "model": str(path), **desc})
            elif req_type == "IrHeader":
                path = msg.get("path") or state.get("model")
                if not path or not es_modelo_ir(path):
                    send("Error", **{"@id": rid,
                                     "message": "no hay un modelo IR cargado"})
                else:
                    send("IrInfo", **{"@id": rid, "model": str(path),
                                      **ir_reader.describe_ir(str(path))})
            elif req_type == "CodeOutline":
                target = _confined_path(msg["path"], workspace)
                source = edit_formats.read_text(target)
                language = msg.get("language") or code_structure.language_for_path(target)
                symbols = [
                    {"kind": s.kind, "qualified_name": s.qualified_name,
                     "start_line": s.start_line, "end_line": s.end_line}
                    for s in code_structure.list_symbols(source, language)
                ]
                send("Outline", **{"@id": rid, "path": str(target),
                                   "outline": code_structure.outline(source, language),
                                   "symbols": symbols})
            elif req_type == "ExtractSymbol":
                target = _confined_path(msg["path"], workspace)
                source = edit_formats.read_text(target)
                language = msg.get("language") or code_structure.language_for_path(target)
                code = code_structure.extract_symbol(source, msg["symbol"], language)
                send("Symbol", **{"@id": rid, "path": str(target),
                                  "symbol": msg["symbol"], "code": code})
            elif req_type == "McpListServers":
                send("McpServers", **{"@id": rid, "items": sorted(mcp_config.keys())})
            elif req_type == "McpListTools":
                client = get_mcp(msg["server"])
                send("McpTools", **{"@id": rid, "server": msg["server"],
                                    "tools": client.list_tools()})
            elif req_type == "McpCallTool":
                client = get_mcp(msg["server"])
                result = client.call_tool(msg["tool"], msg.get("arguments") or {})
                send("McpResult", **{"@id": rid, "server": msg["server"],
                                     "tool": msg["tool"], "result": result})
            elif req_type == "Ping":
                send("Pong", **{"@id": rid})
            elif req_type == "Quit":
                break
            else:
                send("Error", **{"@id": rid, "message": f"@type desconocido: {req_type}"})
        except SystemExit as exc:
            send("Error", **{"@id": rid, "message": f"descarga/carga falló: {exc}"})
        except Exception as exc:  # noqa: BLE001 - informa de cualquier fallo a la UI
            send("Error", **{"@id": rid, "message": str(exc)})

    for client in mcp_clients.values():
        client.close()
    return 0


# --------------------------------------------------------------------------- #
# Modo interactivo (TUI estilo Claude Code)
# --------------------------------------------------------------------------- #

BANNER = r"""
 _     _
| |   | |                         __     __  ___   _   _    ___
| |   | | __ _ _ __ ___   __ _    \ \   / / |_ _| | \ | |  / _ \
| |   | |/ _` | '_ ` _ \ / _` |    \ \ / /   | |  |  \| | | | | |
| |___| | (_| | | | | | | (_| |     \ V /    | |  | |\  | | |_| |
|______|_|\__,_|_| |_| |_|\__,_|     \_/    |___| |_| \_|  \___/
"""

# Autoría mostrada bajo el banner.
AUTOR = "(c) Rafael Ausejo Prieto, con ayuda de Claude Code"


def scan_local_models(models_dir: Path) -> list[Path]:
    """Devuelve los .gguf en ``<models_dir>/gguf`` (y en la raíz, compatibilidad)."""
    base = Path(models_dir)
    vistos: dict[str, Path] = {}
    for carpeta in (dir_gguf(base), base):
        if carpeta.is_dir():
            for f in sorted(carpeta.glob("*.gguf")):
                vistos.setdefault(f.name, f)
    return list(vistos.values())


def scan_local_ir(models_dir: Path) -> list[Path]:
    """Devuelve los directorios IR en ``<models_dir>/ir`` (y en la raíz)."""
    base = Path(models_dir)
    vistos: dict[str, Path] = {}
    for carpeta in (dir_ir(base), base):
        if carpeta.is_dir():
            for d in sorted(carpeta.iterdir()):
                if es_modelo_ir(d):
                    vistos.setdefault(d.name, d)
    return list(vistos.values())


def _ruta_registro(models_dir: Path, info: dict) -> tuple[Path, bool]:
    """Resuelve la ruta local de una entrada del registro y si está descargada.

    Busca en el subdirectorio nuevo (gguf/ir) y, por compatibilidad, en la raíz.
    """
    base = Path(models_dir)
    if info.get("format") == "ir":
        candidatos = [dir_ir(base) / info["dir"], base / info["dir"]]
        for c in candidatos:
            if es_modelo_ir(c):
                return c, True
        return candidatos[0], False
    candidatos = [dir_gguf(base) / info["file"], base / info["file"]]
    for c in candidatos:
        if c.is_file():
            return c, True
    return candidatos[0], False


def _tam_dir(path: Path) -> str:
    """Tamaño total aproximado (GB) de los ficheros de un directorio."""
    total = sum(f.stat().st_size for f in path.glob("*") if f.is_file())
    return f"{total / 1e9:.2f} GB"


def _tam_gb(size_str: str) -> float:
    """Extrae los GB numéricos de una cadena de tamaño (p. ej. '~8.5 GB' -> 8.5)."""
    match = re.search(r"[\d.]+", size_str or "")
    return float(match.group(0)) if match else 0.0


def _build_model_menu(models_dir: Path) -> list[dict]:
    """Combina el registro con los modelos locales (GGUF e IR) en filas de menú.

    Cada fila: {name, size, format ("GGUF"/"IR"), note, path (Path|None),
               target (alias|None), downloaded (bool), engine}. Ordenadas por
               tamaño descendente.
    """
    perf = perf_db.todos()  # rendimiento medido por nombre de modelo

    def _añadir(row):
        registro = perf.get(Path(row["path"]).name) if row["path"] else None
        row["tok_s"] = registro.get("tok_s") if registro else None
        row["load_seconds"] = registro.get("load_seconds") if registro else None
        rows.append(row)

    rows: list[dict] = []
    seen_files: set[str] = set()
    seen_dirs: set[str] = set()
    for alias, info in MODELS.items():
        formato = "IR" if info.get("format") == "ir" else "GGUF"
        local, downloaded = _ruta_registro(models_dir, info)
        if downloaded:
            (seen_dirs if formato == "IR" else seen_files).add(local.name)
        _añadir(
            {
                "name": alias,
                "size": info["size"],
                "format": formato,
                "publisher": info["repo"].split("/")[0],
                "highlight": info.get("highlight", False),
                "recomendacion": RECOMENDACION.get(alias),
                "note": info["note"],
                "path": local if downloaded else None,
                "target": alias,
                "downloaded": downloaded,
                "engine": info.get("engine", "auto"),
            }
        )
    # Cualquier GGUF local que no esté en el registro (p. ej. del usuario).
    for path in scan_local_models(models_dir):
        if path.name in seen_files:
            continue
        _añadir(
            {
                "name": path.name,
                "size": f"{path.stat().st_size / 1e9:.2f} GB",
                "format": "GGUF",
                "publisher": "local",
                "highlight": False,
                "note": "Fichero local.",
                "path": path,
                "target": None,
                "downloaded": True,
                "engine": _detectar_motor_gguf(path),
            }
        )
    # Directorios de modelo OpenVINO IR locales no listados ya por el registro.
    for path in scan_local_ir(models_dir):
        if path.name in seen_dirs:
            continue
        _añadir(
            {
                "name": path.name + "/",
                "size": _tam_dir(path),
                "format": "IR",
                "publisher": "local",
                "highlight": False,
                "note": "Modelo OpenVINO IR (mmap del .bin).",
                "path": path,
                "target": None,
                "downloaded": True,
                "engine": "openvino",
            }
        )
    # Ordena por prioridad de recomendación (1, 2, 3...); el resto, por tamaño desc.
    rows.sort(key=lambda r: (r.get("recomendacion") or 9999, -_tam_gb(r["size"])))
    return rows


# Color por publicador para la columna "Autor" del selector.
_AUTOR_COLOR = {
    "bartowski": "cyan", "OpenVINO": "green", "unsloth": "magenta",
    "TheBloke": "yellow", "MaziyarPanahi": "blue", "local": "dim",
}


def etiqueta_autor(publisher: str) -> str:
    """Texto con marcado rich para la columna de autor/publicador."""
    color = _AUTOR_COLOR.get(publisher, "white")
    return f"[{color}]{publisher}[/]"


def _picker_lista(lineas, *, titulo="", get_help=None, idx_inicial=0, acciones=None):
    """Selector genérico con flechas (prompt_toolkit). Devuelve el índice o None.

    ``lineas`` es una lista de cadenas (una por opción); ``get_help(i)`` opcional
    devuelve la ayuda de la opción resaltada (se muestra debajo). Navega con
    ↑/↓ (y RePág/AvPág), selecciona con Enter, cancela con q/Esc/Ctrl-C. La línea
    resaltada se sigue al desplazarse (la ventana hace scroll en listas largas).

    ``acciones`` opcional: ``{tecla: (nombre, descripción)}``; al pulsar esa tecla
    el picker devuelve ``(nombre, índice)`` en vez de un entero.
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.document import Document
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.styles import Style

    if not lineas:
        return None
    buf = Buffer(document=Document("\n".join(lineas), 0), read_only=True)
    for _ in range(max(0, min(idx_inicial, len(lineas) - 1))):
        buf.cursor_down()

    lista = Window(BufferControl(buffer=buf, focusable=True), cursorline=True,
                   wrap_lines=False)
    partes = []
    if titulo:
        partes.append(Window(FormattedTextControl(lambda: [("bold cyan", titulo)]),
                             height=1))
    partes.append(lista)
    if get_help is not None:
        partes.append(Window(
            FormattedTextControl(lambda: [("class:ayuda",
                                           get_help(buf.document.cursor_position_row))]),
            height=Dimension(min=1, max=4), wrap_lines=True, style="class:ayuda"))
    extra_pista = ""
    if acciones:
        extra_pista = " · " + " · ".join(f"{t} {d}" for t, (_, d) in acciones.items())
    partes.append(Window(FormattedTextControl(
        lambda: [("class:pista",
                  "↑/↓ moverse · Enter seleccionar · q/Esc cancelar" + extra_pista)]),
        height=1))

    kb = KeyBindings()
    kb.add("up")(lambda e: e.app.current_buffer.cursor_up())
    kb.add("down")(lambda e: e.app.current_buffer.cursor_down())
    kb.add("pageup")(lambda e: e.app.current_buffer.cursor_up(10))
    kb.add("pagedown")(lambda e: e.app.current_buffer.cursor_down(10))
    kb.add("enter")(lambda e: e.app.exit(result=e.app.current_buffer.document.cursor_position_row))
    for tecla in ("q", "escape", "c-c"):
        kb.add(tecla)(lambda e: e.app.exit(result=None))
    for tecla, (nombre, _desc) in (acciones or {}).items():
        def _hacer(e, _n=nombre):
            e.app.exit(result=(_n, e.app.current_buffer.document.cursor_position_row))
        kb.add(tecla)(_hacer)

    estilo = Style.from_dict({
        "cursor-line": "reverse", "ayuda": "fg:#9e9e9e", "pista": "fg:#666666",
    })
    app = Application(layout=Layout(HSplit(partes), focused_element=lista),
                     key_bindings=kb, style=estilo, full_screen=True,
                     mouse_support=False)  # navegación solo con teclado (flechas)
    return app.run()


def _fila_modelo(row: dict) -> str:
    """Línea de texto plano (columnas alineadas) de un modelo para el picker."""
    marca = "✦" if row.get("highlight") else " "
    estado = "✓" if row["downloaded"] else "⬇"
    rec = f"#{row['recomendacion']}" if row.get("recomendacion") else " ·"
    rend = ""
    if row.get("tok_s"):
        rend += f"{row['tok_s']:.0f}t/s"
    if row.get("load_seconds"):
        rend += f" {row['load_seconds']:.0f}s"
    return (f"{marca}{rec:>3} {row['name'][:26]:<26} {row.get('publisher', '')[:12]:<12} "
            f"{row['size']:>9}  {row.get('format', 'GGUF'):<5} "
            f"{row.get('engine', 'auto'):<9} {estado}  {rend:<12}")


def interactive_select_model(console, args):
    """Selector de modelos con flechas; devuelve una ruta lista (descargada) o None."""
    from rich.prompt import Prompt
    from rich.table import Table

    ultimo = cargar_ajustes().get("last_model")  # ruta del último modelo usado

    def _recordar(path):
        guardar_ajustes(last_model=str(path))
        return path

    def _picker_numerico(rows, idx_ultimo):  # respaldo si no hay consola TUI
        table = Table(title="Modelos disponibles", title_style="bold cyan", expand=True)
        for col in ("#", "Recom.", "Modelo", "Autor", "Tamaño", "Formato", "Motor",
                    "Estado", "Descripción"):
            table.add_column(col)
        for idx, row in enumerate(rows, 1):
            status = "[green]✓[/]" if row["downloaded"] else "[yellow]⬇[/]"
            rec = f"[bold green]{row['recomendacion']}[/]" if row.get("recomendacion") else ""
            table.add_row(str(idx), rec, row["name"],
                          etiqueta_autor(row.get("publisher", "")),
                          row["size"], row.get("format", "GGUF"),
                          etiqueta_motor(row.get("engine", "auto")), status, row["note"])
        console.print(table)
        choice = Prompt.ask("Modelo (número o 'q')", default="").strip().lower()
        if choice in {"q", "quit", "salir", "exit"}:
            return None
        if choice == "" and idx_ultimo is not None:
            return idx_ultimo
        return int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= len(rows) else -1

    acciones = {
        "b": ("buscar", "buscar nuevos"),
        "d": ("borrar", "borrar del disco"),
    }
    while True:
        rows = _build_model_menu(args.models_dir)
        idx_ultimo = next(
            (i for i, r in enumerate(rows)
             if r["downloaded"] and r["path"] and str(r["path"]) == ultimo), None)
        lineas = []
        for i, r in enumerate(rows):
            marca = " (último)" if i == idx_ultimo else ""
            lineas.append(_fila_modelo(r) + marca)
        titulo = ("Rec Modelo                     Autor          Tamaño  Form. Motor    "
                  " Est. Rend.  (#=recomendado, ✦ unsloth)")
        try:
            sel = _picker_lista(lineas, titulo=titulo,
                                get_help=lambda i: rows[i]["note"],
                                idx_inicial=idx_ultimo or 0, acciones=acciones)
        except Exception:  # noqa: BLE001 - sin consola TUI: respaldo numérico
            sel = _picker_numerico(rows, idx_ultimo)
        if sel is None:
            return None
        # Acción (borrar/buscar) -> (nombre, índice).
        if isinstance(sel, tuple):
            accion, idx = sel
            if accion == "buscar":
                nuevo = _buscar_y_descargar(console, args)
                if nuevo is not None:
                    return _recordar(nuevo)
                continue
            if accion == "borrar" and 0 <= idx < len(rows):
                _borrar_interactivo(console, rows[idx])
            continue
        if not (0 <= sel < len(rows)):
            console.print("[red]Opción no válida.[/]")
            continue

        row = rows[sel]
        if not row["downloaded"]:
            console.print(f"Descargando [cyan]{row['name']}[/] ({row['size']}) ...")
            try:
                path = download_model(row["target"], args.models_dir)
            except SystemExit:
                console.print("[red]Descarga cancelada o fallida.[/]")
                continue
        else:
            path = row["path"]
        return _recordar(path)


def _borrar_interactivo(console, row) -> None:
    """Borra del disco el modelo de ``row`` tras confirmar (acción 'd')."""
    from rich.prompt import Confirm

    if not row["downloaded"] or not row["path"]:
        console.print("[yellow]Ese modelo no está descargado.[/]")
        return
    if Confirm.ask(f"¿Borrar del disco [cyan]{row['name']}[/] "
                   f"({row['size']})?", default=False):
        try:
            borrar_modelo_disco(row["path"])
            console.print(f"[green]✓ Borrado:[/] {row['path']}")
        except OSError as exc:
            console.print(f"[red]No se pudo borrar: {exc}[/]")


def _buscar_y_descargar(console, args):
    """Busca modelos en HF por filtros, deja elegir uno y lo descarga (acción 'b').

    Devuelve la ruta descargada o None si se cancela / falla.
    """
    from rich.prompt import Prompt

    formato = Prompt.ask("Formato", choices=["gguf", "ir"], default="gguf")
    consulta = Prompt.ask("Texto a buscar (p. ej. 'Qwen2.5')", default="")
    rango = Prompt.ask("Tamaño en B (min-max)", default="1-14")
    try:
        min_b, max_b = (float(x) for x in rango.split("-", 1))
    except ValueError:
        min_b, max_b = 0.0, 14.0
    console.print("[dim]Buscando en Hugging Face…[/]")
    try:
        candidatos = buscar_modelos(formato=formato, consulta=consulta,
                                    min_b=min_b, max_b=max_b)
    except Exception as exc:  # noqa: BLE001 - red/HF puede fallar
        console.print(f"[red]Búsqueda fallida: {exc}[/]")
        return None
    if not candidatos:
        console.print("[yellow]Sin resultados.[/]")
        return None

    lineas = [f"{c['repo']:<55} {(str(c['params_b']) + 'B') if c['params_b'] else '?':>5}"
              for c in candidatos]
    try:
        idx = _picker_lista(lineas, titulo=f"Resultados ({formato}) — Enter descarga",
                            get_help=lambda i: candidatos[i]["repo"])
    except Exception:  # noqa: BLE001
        idx = None
    if idx is None or isinstance(idx, tuple) or not (0 <= idx < len(candidatos)):
        return None

    repo = candidatos[idx]["repo"]
    try:
        if formato == "ir":
            return _descargar_ir(repo, dir_ir(args.models_dir) / repo.split("/")[-1])
        cuant = Prompt.ask("Cuantización", default="Q4_K_M")
        ficheros = _ficheros_gguf(repo, cuant)
        if not ficheros:
            console.print("[yellow]No hay .gguf en ese repo.[/]")
            return None
        return download_model(f"{repo}:{ficheros[0]}", args.models_dir)
    except (SystemExit, Exception) as exc:  # noqa: BLE001 - descarga puede fallar
        console.print(f"[red]Descarga fallida: {exc}[/]")
        return None


def _format_setting_value(key: str, value) -> str:
    """Formatea el valor de un ajuste para la tabla de configuración."""
    if key == "stop_strings":
        return ", ".join(value) if value else "[dim](ninguna)[/]"
    return str(value)


def _show_settings_table(console, settings: dict) -> None:
    """Imprime los ajustes de generación actuales como una tabla rich."""
    from rich.table import Table

    table = Table(title="Configuración de generación", title_style="bold cyan",
                  expand=True)
    table.add_column("#", justify="right", style="bold")
    table.add_column("Parámetro", style="cyan")
    table.add_column("Valor", justify="right", style="green")
    table.add_column("Descripción", style="dim")
    for idx, (key, spec) in enumerate(GEN_PARAM_SPECS.items(), 1):
        table.add_row(
            str(idx), spec["label"],
            _format_setting_value(key, settings[key]), spec["help"],
        )
    console.print(table)


def _coerce_setting(key: str, raw: str):
    """Convierte y valida una cadena en el valor tipado de ``key``.

    Raises:
      ValueError: Si el valor no se puede parsear o está fuera de rango.
    """
    spec = GEN_PARAM_SPECS[key]
    kind = spec["type"]
    if kind is list:
        # Secuencias de parada separadas por comas; vacío las borra.
        return [s.strip() for s in raw.split(",") if s.strip()]
    value = kind(raw)
    lo, hi = spec.get("min"), spec.get("max")
    if lo is not None and value < lo or hi is not None and value > hi:
        raise ValueError(f"fuera de rango [{lo}, {hi}]")
    return value


def _texto_valor_plano(key: str, value) -> str:
    """Como ``_format_setting_value`` pero en texto plano (para el picker)."""
    if key == "stop_strings":
        return ", ".join(value) if value else "(ninguna)"
    return str(value)


def configure_settings(console, settings: dict) -> None:
    """Editor de parámetros de generación con selección por flechas.

    Modifica ``settings`` in situ. Cada parámetro muestra su **ayuda** al
    resaltarlo; Enter lo edita (se teclea el valor) y q/Esc vuelve al chat.
    """
    from rich.prompt import Prompt

    keys = list(GEN_PARAM_SPECS.keys())

    def _numerico() -> int | None:  # respaldo sin consola TUI
        _show_settings_table(console, settings)
        choice = Prompt.ask("Parámetro (# o 'q')", default="").strip().lower()
        if choice in {"q", "quit", "salir", "exit", "done", "ok", ""}:
            return None
        return int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= len(keys) else -1

    while True:
        lineas = [f"{GEN_PARAM_SPECS[k]['label']:<24} = "
                  f"{_texto_valor_plano(k, settings[k])}" for k in keys]
        try:
            sel = _picker_lista(
                lineas, titulo="Configuración de generación (Enter edita)",
                get_help=lambda i: GEN_PARAM_SPECS[keys[i]]["help"])
        except Exception:  # noqa: BLE001 - respaldo numérico
            sel = _numerico()
        if sel is None:
            break
        if not (0 <= sel < len(keys)):
            console.print("[red]Opción no válida.[/]")
            continue

        key = keys[sel]
        spec = GEN_PARAM_SPECS[key]
        console.print(f"\n[cyan]{spec['label']}[/] — [dim]{spec['help']}[/]")
        current = _format_setting_value(key, settings[key])
        if spec["type"] is list:
            console.print(f"[dim]Actual: {current}. Separa varias con comas; "
                          "vacío para borrar.[/]")
        raw = Prompt.ask(f"Nuevo valor para [cyan]{spec['label']}[/]")
        try:
            settings[key] = _coerce_setting(key, raw)
        except ValueError as exc:
            console.print(f"[red]Valor no válido: {exc}[/]")
            continue
        console.print(
            f"[green]✓[/] {spec['label']} = "
            f"{_format_setting_value(key, settings[key])}"
        )


# Comandos «/» disponibles en el REPL del chat. Reproducen el conjunto de Claude
# Code, adaptados a un motor local GGUF + OpenVINO. ``names`` lista todas las
# grafías/alias aceptados; el primero es el canónico que se muestra en la ayuda.
CHAT_COMMANDS: list[dict] = [
    {"names": ["/help", "/ayuda", "/?"], "help": "Muestra esta ayuda."},
    {"names": ["/clear", "/limpiar"],
     "help": "Reinicia la conversación (borra el contexto)."},
    {"names": ["/compact", "/compactar"],
     "help": "Resume la conversación para ahorrar contexto."},
    {"names": ["/config", "/configuration", "/configuración", "/ajustes"],
     "help": "Ajusta temperatura, top-p/k, penalizaciones, tokens y parada."},
    {"names": ["/model", "/models", "/modelo"], "help": "Cambia de modelo."},
    {"names": ["/save", "/guardar"],
     "help": "Guarda el último código generado: /save [ruta]."},
    {"names": ["/cost", "/coste"],
     "help": "Uso de tokens y tiempos de la sesión."},
    {"names": ["/status", "/estado"],
     "help": "Modelo, dispositivo, motor y ajustes activos."},
    {"names": ["/gguf"],
     "help": "Cabecera del GGUF: /gguf [filtro] · /gguf tokens [N] vuelca arrays."},
    {"names": ["/ir"],
     "help": "Información del modelo OpenVINO IR cargado (ficheros, config, rt_info)."},
    {"names": ["/gpu", "/offload"],
     "help": "Reparto de capas GPU/CPU calculado en tiempo real (mmap)."},
    {"names": ["/color"],
     "help": "Color de las barras: /color [red|blue|green|yellow|purple|"
             "orange|pink|cyan|default]."},
    {"names": ["/doctor"], "help": "Diagnóstico de OpenVINO y dispositivos."},
    {"names": ["/mcp"], "help": "Lista los servidores MCP configurados."},
    {"names": ["/candidatas", "/candidates"],
     "help": "Panel de candidatas (llama-server): /candidatas [on|off] alterna la "
             "vista en vivo · /candidatas historico revisa lo registrado."},
    {"names": ["/exit", "/quit", "/salir"], "help": "Sale de LlamaVino."},
]


# Paleta para el comando /color: nombre -> color (hex de 256). "default" reinicia.
COLOR_OPCIONES = ["red", "blue", "green", "yellow", "purple", "orange",
                  "pink", "cyan", "default"]
COLOR_HEX = {
    "red": "#d75f5f", "blue": "#5f87af", "green": "#5faf5f", "yellow": "#d7af5f",
    "purple": "#8787d7", "orange": "#d78700", "pink": "#d787af", "cyan": "#5fd7d7",
    "default": "#5f87af",
}
# Azul para resaltar los comandos «/» (en el input y en el menú).
COMANDO_AZUL = "#5fafff"


def _resolve_command(text: str) -> str | None:
    """Devuelve el nombre canónico del comando de ``text``, o None si se desconoce."""
    token = text.split()[0].lower()
    for cmd in CHAT_COMMANDS:
        if token in cmd["names"]:
            return cmd["names"][0]
    return None


def _print_help(console) -> None:
    """Imprime la referencia de comandos del chat como una tabla rich."""
    from rich.table import Table

    table = Table(title="Comandos", title_style="bold cyan", expand=True)
    table.add_column("Comando", style="cyan", no_wrap=True)
    table.add_column("Alias", style="dim", no_wrap=True)
    table.add_column("Descripción")
    for cmd in CHAT_COMMANDS:
        canonical = cmd["names"][0]
        aliases = ", ".join(cmd["names"][1:]) or "—"
        table.add_row(canonical, aliases, cmd["help"])
    console.print(table)
    console.print(
        "[dim]Atajos: al teclear «/» aparece el menú (↑/↓ elegir, Tab completar, "
        "Enter ejecutar). Ratón/rueda para desplazar la salida; Ctrl-C cancela, "
        "Ctrl-D sale. Los comandos largos se ven con ENTER para volver.[/]"
    )


def _print_status(console, model_path, engine: dict, settings) -> None:
    """Muestra el modelo, dispositivo, motor y ajustes de generación activos."""
    from rich.table import Table

    table = Table(title="Estado", title_style="bold cyan", show_header=False)
    table.add_column("k", style="cyan")
    table.add_column("v")
    name = Path(model_path).name if model_path else "(ninguno)"
    table.add_row("Modelo", name)
    table.add_row("Dispositivo", str(engine.get("device", "?")))
    table.add_row("Motor", str(engine.get("kind", "?")))
    if engine.get("n_gpu_layers") is not None:
        plan = engine.get("plan")
        total = f"/{plan.block_count}" if plan else ""
        table.add_row("Capas en GPU", f"{engine['n_gpu_layers']}{total} (resto en CPU)")
    table.add_row("Temperatura", str(settings["temperature"]))
    table.add_row("top_p / top_k", f"{settings['top_p']} / {settings['top_k']}")
    table.add_row(
        "Penalizaciones",
        f"presencia {settings['presence_penalty']} · "
        f"frecuencia {settings['frequency_penalty']}",
    )
    table.add_row("Máx. tokens", str(settings["max_new_tokens"]))
    stops = ", ".join(settings["stop_strings"]) or "(ninguna)"
    table.add_row("Parada", stops)
    console.print(table)


def _print_cost(console, stats: dict) -> None:
    """Muestra el uso de tokens y tiempos de la sesión (análogo local de /cost)."""
    from rich.table import Table

    secs = stats["seconds"]
    toks = stats["output_tokens"]
    tps = toks / secs if secs > 0 else 0.0
    table = Table(title="Uso de la sesión", title_style="bold cyan",
                  show_header=False)
    table.add_column("k", style="cyan")
    table.add_column("v", justify="right")
    table.add_row("Turnos", str(stats["turns"]))
    table.add_row("Tokens generados", str(toks))
    table.add_row("Tiempo de generación", f"{secs:.1f} s")
    table.add_row("Velocidad media", f"{tps:.1f} tok/s")
    console.print(table)
    console.print("[dim]Modelo local: sin coste monetario (cómputo propio).[/]")


def _run_doctor(console) -> None:
    """Diagnostica la instalación de OpenVINO y los dispositivos (análogo de /doctor)."""
    from rich.table import Table

    try:
        import openvino as ov
    except ImportError as exc:  # pragma: no cover - depende del entorno
        console.print(f"[red]OpenVINO no disponible: {exc}[/]")
        return
    core = ov.Core()
    devices = core.available_devices
    has_gpu = any(d.startswith("GPU") for d in devices)
    table = Table(title="Diagnóstico OpenVINO", title_style="bold cyan",
                  show_header=False)
    table.add_column("k", style="cyan")
    table.add_column("v")
    table.add_row("Versión OpenVINO", getattr(ov, "__version__", "?"))
    table.add_row("Dispositivos", ", ".join(devices) or "(ninguno)")
    table.add_row("GPU detectada", "[green]sí[/]" if has_gpu else "[yellow]no[/]")
    for name in devices:
        try:
            full = core.get_property(name, "FULL_DEVICE_NAME")
        except Exception:  # noqa: BLE001 - la propiedad es opcional por dispositivo
            full = "<desconocido>"
        table.add_row(name, str(full))
    console.print(table)


def _gib(num_bytes: float) -> str:
    """Formatea bytes como GiB legibles."""
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def _show_ir(console, model_path) -> None:
    """Muestra la información de un modelo OpenVINO IR (equivalente a /gguf).

    Lista los ficheros y tamaños, la configuración del modelo (``config.json``)
    con su significado, la configuración de generación y el ``rt_info`` del XML.
    """
    from rich.table import Table

    if not model_path:
        console.print("[yellow]No hay ningún modelo cargado.[/]")
        return
    if not es_modelo_ir(model_path):
        console.print("[dim]El modelo cargado no es OpenVINO IR (es un GGUF); "
                      "usa [cyan]/gguf[/].[/]")
        return
    try:
        info = ir_reader.describe_ir(str(model_path))
    except (OSError, ValueError) as exc:
        console.print(f"[red]No se pudo leer el modelo IR: {exc}[/]")
        return

    console.print(
        f"[bold cyan]Modelo OpenVINO IR[/] · {info['name']} · "
        f"{_gib(info['total_bytes'])} · arquitectura: "
        f"[cyan]{info['architecture']}[/]"
    )

    ficheros = Table(title="Ficheros", title_style="bold", expand=True)
    ficheros.add_column("Fichero", style="cyan")
    ficheros.add_column("Tamaño", justify="right", style="green")
    for f in info["files"]:
        ficheros.add_row(f["name"], _gib(f["bytes"]))
    console.print(ficheros)

    if info["config_rows"]:
        cfg = Table(title="Configuración (config.json)", title_style="bold cyan",
                    expand=True)
        cfg.add_column("Parámetro", style="cyan", no_wrap=True)
        cfg.add_column("Valor", style="green")
        cfg.add_column("Significado", style="dim")
        for row in info["config_rows"]:
            cfg.add_row(row["key"], row["value"], row["meaning"])
        console.print(cfg)

    if info["rt_info"]:
        rt = Table(title="rt_info (OpenVINO/optimum)", title_style="bold cyan",
                   show_header=False)
        rt.add_column("k", style="cyan")
        rt.add_column("v", style="green")
        for row in info["rt_info"]:
            rt.add_row(row["key"], row["value"])
        console.print(rt)

    if info["gen_rows"]:
        gen = Table(title="generation_config.json", title_style="bold cyan",
                    show_header=False)
        gen.add_column("k", style="cyan")
        gen.add_column("v", style="green")
        for row in info["gen_rows"]:
            gen.add_row(row["key"], row["value"])
        console.print(gen)


def _show_gguf(console, model_path, key_filter: str | None = None,
               array_limit: int | None = None) -> None:
    """Lee la cabecera del GGUF cargado y muestra sus parámetros y significado.

    Con ``key_filter`` sólo se muestran las claves coincidentes; los arrays que
    coincidan (p. ej. ``tokens``) se vuelcan completos en vez de resumirse.
    ``array_limit`` acota cuántos elementos se vuelcan por array.
    """
    from rich.columns import Columns
    from rich.markup import escape
    from rich.table import Table

    if not model_path:
        console.print("[yellow]No hay ningún modelo cargado.[/]")
        return
    if es_modelo_ir(model_path):
        console.print("[dim]El modelo cargado es OpenVINO IR, no un GGUF; "
                      "/gguf no aplica.[/]")
        return
    try:
        info = gguf_reader.describe_gguf(str(model_path), key_filter=key_filter,
                                         array_limit=array_limit)
    except (OSError, ValueError) as exc:
        console.print(f"[red]No se pudo leer la cabecera GGUF: {exc}[/]")
        return

    suffix = (f" · filtro: [yellow]{key_filter}[/] "
              f"({info['matched']}/{info['total']} claves)") if key_filter else ""
    console.print(
        f"[bold cyan]Cabecera GGUF[/] · v{info['version']} · "
        f"{info['tensor_count']} tensores · {info['kv_count']} claves · "
        f"arquitectura: [cyan]{info['architecture']}[/]{suffix}"
    )
    if not info["rows"]:
        console.print(f"[yellow]Sin coincidencias para '{key_filter}'.[/]")
        return

    table = Table(title=Path(model_path).name, title_style="bold", expand=True)
    table.add_column("Parámetro", style="cyan", no_wrap=True)
    table.add_column("Valor", style="green")
    table.add_column("Significado", style="dim")
    for row in info["rows"]:
        table.add_row(row["key"], row["value"], row["meaning"])
    console.print(table)

    # Vuelca el contenido de los arrays coincidentes (p. ej. tokens).
    for row in info["rows"]:
        if not row["array"]:
            continue
        arr = row["array"]
        values = arr["values"]
        total = arr["len"]
        nota = f" (mostrando {len(values)} de {total})" if len(values) < total else ""
        console.print(f"\n[bold]{row['key']}[/] — {total} elementos{nota}:")
        console.print(Columns([escape(str(v)) for v in values], padding=(0, 2)))


def _show_gpu_plan(console, engine: dict, model_path, device, args) -> None:
    """Muestra el reparto de capas GPU/CPU recalculado en tiempo real (/gpu)."""
    from rich.table import Table

    if not model_path:
        console.print("[yellow]No hay ningún modelo cargado.[/]")
        return
    try:
        plan = layer_planner.plan_offload(
            str(model_path), device=device, n_ctx=int(getattr(args, "n_ctx", 4096)))
    except (OSError, ValueError) as exc:
        console.print(f"[red]No se pudo calcular el reparto: {exc}[/]")
        return

    table = Table(title="Reparto de capas (GPU/CPU)", title_style="bold cyan",
                  show_header=False)
    table.add_column("k", style="cyan")
    table.add_column("v")
    table.add_row("Capas totales", str(plan.block_count))
    table.add_row("→ a la GPU (VRAM)", str(plan.n_gpu_layers))
    table.add_row("→ a la CPU (RAM)", str(plan.block_count - plan.n_gpu_layers))
    table.add_row("Por capa (pesos+KV)", layer_planner.gib(plan.per_layer_bytes))
    table.add_row("VRAM total", layer_planner.gib(plan.vram_total))
    table.add_row("VRAM utilizable", layer_planner.gib(plan.vram_usable))
    table.add_row("RAM libre", layer_planner.gib(plan.free_ram))
    table.add_row("Tamaño del modelo", layer_planner.gib(plan.file_size))
    console.print(table)
    console.print(f"[dim]{plan.reason}[/]")

    motor = engine.get("kind")
    if motor == "llamacpp":
        console.print(
            f"[green]Motor activo: llama.cpp con -ngl "
            f"{engine.get('n_gpu_layers')} y mmap.[/]"
        )
    else:
        aviso = ("Motor activo: OpenVINO (carga todo en el dispositivo). Para "
                 "repartir capas con mmap, reinicia con [cyan]--engine llamacpp[/] "
                 "o [cyan]--engine auto[/]")
        if not _hay_binario_llamacpp():
            aviso += " (requiere el binario de llama.cpp; ver LLAMA_CPP_BIN)"
        console.print(f"[dim]{aviso}.[/]")


def _tabla_candidatos(cands: list[dict], elegida: str):
    """Tabla rich con las palabras candidatas del token y la elegida resaltada.

    La palabra elegida se muestra en verde con «→»; el resto, atenuadas. Se usa
    ``repr`` para que se vean espacios y saltos de línea de cada token.
    """
    from rich.table import Table
    from rich.text import Text

    tabla = Table(title="Candidatas para el próximo token (n_probs)",
                  title_style="bold cyan", show_edge=False, expand=False)
    tabla.add_column("", width=2)
    tabla.add_column("Palabra")
    tabla.add_column("Prob.", justify="right")
    for cand in cands:
        es_elegida = cand["word"] == elegida
        estilo = "bold green" if es_elegida else "dim"
        tabla.add_row(
            Text("→" if es_elegida else "", style=estilo),
            Text(repr(cand["word"]), style=estilo),
            Text(f"{cand['prob'] * 100:5.1f}%", style=estilo),
        )
    return tabla


def _print_candidatas_hist(console, hist: list[dict]) -> None:
    """Imprime el histórico de candidatas por token registrado en una generación.

    Una línea por token: posición, palabra elegida (verde) con su probabilidad y las
    demás candidatas atenuadas. Pensado para revisarlo en el pager tras generar.
    """
    from rich.text import Text

    console.print(f"[bold cyan]Histórico de candidatas — {len(hist)} tokens "
                  "(n_probs)[/bold cyan]\n")
    for e in hist:
        elegida = e["elegida"]
        prob_el = next((c["prob"] for c in e["lista"] if c["word"] == elegida), None)
        linea = Text()
        linea.append(f"#{e['pos']:>4}  ", style="dim")
        linea.append(f"→ {elegida!r}", style="bold green")
        if prob_el is not None:
            linea.append(f" {prob_el * 100:.1f}%", style="green")
        alts = [c for c in e["lista"] if c["word"] != elegida]
        if alts:
            linea.append("   alt: ", style="dim")
            partes = [f"{c['word']!r} {c['prob'] * 100:.1f}%" for c in alts[:6]]
            linea.append(" · ".join(partes), style="#808080")
        console.print(linea)


def _cmd_color(console, tema: dict, arg: str) -> None:
    """Cambia el color de las barras del prompt (/color)."""
    opcion = (arg or "").strip().lower()
    if opcion not in COLOR_HEX:
        opciones = "|".join(COLOR_OPCIONES)
        console.print(f"[yellow]Uso: /color [{opciones}][/]")
        return
    tema["color"] = COLOR_HEX[opcion]
    console.print(f"[dim]Color de las barras: {opcion}.[/]")


def _list_mcp_servers(console, workspace: Path) -> None:
    """Lista los servidores MCP de ``mcp_config.json`` (análogo de /mcp)."""
    from rich.table import Table

    config = _load_mcp_config(workspace)
    if not config:
        console.print(
            "[dim]No hay servidores MCP configurados. Crea "
            "[cyan]mcp_config.json[/] con una clave \"servers\".[/]"
        )
        return
    table = Table(title="Servidores MCP", title_style="bold cyan")
    table.add_column("Nombre", style="cyan")
    table.add_column("Comando")
    for name, entry in sorted(config.items()):
        table.add_row(name, " ".join(entry.get("command", [])) or "—")
    console.print(table)
    console.print("[dim]El tool-loop MCP en el chat aún es roadmap (modelo 1B).[/]")


def _compact_conversation(console, engine, turns, ov_genai):
    """Resume ``turns`` en un único turno de sistema que ahorra contexto.

    Devuelve la nueva lista de turnos. Es el análogo local del /compact de Claude
    Code: pide al motor activo que resuma la transcripción hasta el momento.
    """
    if not turns:
        console.print("[dim]No hay conversación que compactar.[/]")
        return turns
    transcript = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
    prompt = (
        "Resume en español, de forma concisa, la siguiente conversación. "
        "Conserva los hechos, decisiones y contexto importantes para poder "
        "continuar el diálogo más tarde:\n\n" + transcript
    )
    ajustes = {"max_new_tokens": 512, "temperature": 0.0, "top_p": 0.9,
               "top_k": 40, "presence_penalty": 0.0, "frequency_penalty": 0.0,
               "stop_strings": []}
    console.print("[dim]Compactando conversación…[/]")
    try:
        summary, _ = _motor_generar(
            engine, [{"role": "user", "content": prompt}], ajustes, ov_genai,
            lambda _piece: None)
    except KeyboardInterrupt:
        console.print("[dim](compactación interrumpida)[/]")
        return turns
    console.print("[green]✓[/] Conversación compactada.")
    return [{"role": "system",
             "content": "Resumen de la conversación previa:\n" + str(summary)}]


def _confirm_and_write(console, filename: str, code: str, workspace: Path) -> bool:
    """Pregunta al usuario y escribe ``code`` en ``filename`` dentro de ``workspace``.

    Devuelve True si se escribió el fichero. La sobrescritura se confirma aparte
    (por defecto no), siguiendo la regla del harness sobre acciones destructivas.
    """
    from rich.prompt import Confirm

    try:
        target = _confined_path(filename, workspace)
    except edit_formats.PatchError as exc:
        console.print(f"[red]{exc}[/]")
        return False
    exists = target.exists()
    rel = target.relative_to(workspace) if workspace in target.parents else target
    if exists:
        prompt = f"[yellow]{rel}[/] ya existe. ¿Sobrescribir?"
        default = False
    else:
        prompt = f"¿Guardar [cyan]{rel}[/] ({len(code)} bytes)?"
        default = True
    if not Confirm.ask(prompt, default=default):
        console.print("[dim]No guardado.[/]")
        return False
    try:
        written = write_workspace_file(filename, code, workspace)
    except OSError as exc:
        console.print(f"[red]No se pudo escribir {filename}: {exc}[/]")
        return False
    console.print(f"[green]✓ Guardado:[/] {written}")
    return True


def _save_from_response(console, response: str, workspace: Path,
                        explicit_path: str | None = None) -> None:
    """Guarda en disco el código de ``response`` (el gestor local de /save).

    Con ``explicit_path`` el primer bloque de código (o todo el texto) va ahí.
    Si no, se ofrece guardar cada bloque con un nombre de archivo detectado.
    """
    blocks = extract_code_blocks(response)
    if explicit_path:
        code = blocks[0]["code"] if blocks else response
        _confirm_and_write(console, explicit_path, code, workspace)
        return
    named = [b for b in blocks if b["filename"]]
    if not named:
        if blocks:
            console.print(
                "[yellow]Hay código pero no detecté un nombre de archivo.[/] "
                "Usa [cyan]/save <ruta>[/] para guardarlo."
            )
        else:
            console.print("[dim]La última respuesta no contiene código.[/]")
        return
    for block in named:
        _confirm_and_write(console, block["filename"], block["code"], workspace)


def _contexto_modelo(model_path) -> int | None:
    """Devuelve la longitud de contexto (tokens) del modelo, o None si no consta.

    Para GGUF lo lee de ``{arch}.context_length``; para un directorio IR, de
    ``max_position_embeddings`` en su ``config.json``.
    """
    if not model_path:
        return None
    if es_modelo_ir(model_path):
        config = Path(model_path) / "config.json"
        try:
            datos = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        valor = datos.get("max_position_embeddings")
        return int(valor) if isinstance(valor, int) else None
    try:
        meta = gguf_reader.read_gguf_metadata(str(model_path))["metadata"]
    except (OSError, ValueError):
        return None
    arch = meta.get("general.architecture")
    valor = meta.get(f"{arch}.context_length") if arch else None
    return int(valor) if isinstance(valor, int) else None


def _tokens_contexto(turns: list[dict]) -> int:
    """Estima los tokens que ocupa el contexto actual (≈ 4 caracteres/token)."""
    caracteres = len(FILE_PRIMER) + sum(len(t["content"]) for t in turns)
    return caracteres // 4


def _parsear_args_gguf(texto: str) -> tuple[str | None, int | None]:
    """Separa el argumento de ``/gguf`` en (filtro, tope_de_array).

    ``/gguf tokens 100`` -> ("tokens", 100); ``/gguf rope`` -> ("rope", None);
    ``/gguf`` -> (None, None).
    """
    piezas = texto.split()
    limite = None
    if piezas and piezas[-1].isdigit():
        limite = int(piezas[-1])
        piezas = piezas[:-1]
    filtro = " ".join(piezas) or None
    return filtro, limite


class ChatTUI:
    """TUI a pantalla completa del chat (prompt_toolkit).

    Layout, de arriba a abajo: una ventana de **salida** desplazable (la
    conversación y la respuesta en streaming hacen scroll hacia arriba), una
    ventana opcional de **candidatas** (motor llamaserver) y, FIJOS abajo:
    **barra · prompt · barra · estado**. El contador de tokens del estado se
    actualiza en tiempo real durante la generación, que corre en segundo plano
    (executor) para no bloquear la interfaz. La rueda del ratón desplaza la
    salida (``mouse_support``) para ver los mensajes anteriores.
    """

    def __init__(self, console, engine, args, model_path, device, ov_genai,
                 estado=None):
        self.console = console
        self.engine = engine
        self.args = args
        self.model_path = model_path
        self.device = device
        self.ov = ov_genai
        self.workspace = Path.cwd().resolve()
        if estado is not None:
            # Reentrada (tras /model o /config): conserva la conversación.
            self.settings = estado["settings"]
            self.turns = estado["turns"]
            self.stats = estado["stats"]
            self.last_response = estado["last_response"]
            self.tema = estado["tema"]
        else:
            self.settings = default_gen_settings(args)
            self.turns = []
            self.stats = {"turns": 0, "output_tokens": 0, "seconds": 0.0}
            self.last_response = ""
            self.tema = {"color": COLOR_HEX["default"]}
            # Aplica los ajustes persistidos de sesiones anteriores (si los hay).
            _persist = cargar_ajustes()
            self.settings.update({k: v for k, v in _persist.get("settings", {}).items()
                                  if k in self.settings})
            self.tema["color"] = _persist.get("color", self.tema["color"])
        self.ctx_len = _contexto_modelo(model_path)
        self.nombre_modelo = Path(model_path).name if model_path else "(sin modelo)"
        self.ruta_dir = str(self.workspace)
        self.muestra_candidatas = engine.get("kind") == "llamaserver"
        # Vista en vivo del panel de candidatas (alternable con /candidatas; persistida).
        # Por defecto activada. El histórico se registra siempre, esté o no en vivo.
        self.candidatas_vivo = bool(_persist.get("candidatas_vivo", True)) \
            if estado is None else estado.get("candidatas_vivo", True)

        self.transcripcion: list[str] = []   # bloques finalizados (texto plano)
        self.stream_text = ""                # respuesta en curso
        self.generando = False
        self.candidatas = {"lista": [], "elegida": ""}
        # Histórico de candidatas por token de la última generación (revisable con
        # «/candidatas historico»): lista de {pos, elegida, lista}.
        self.hist_candidatas: list[dict] = []
        self.tokens_vivos = 0
        self.accion = None                   # "switch" | None al salir
        self._ctx_base = 0
        self._n_tok = 0
        self._inicio = 0.0
        self._loop = None
        self._cancelar = threading.Event()   # cancelación de la generación (Ctrl-C)

        self._construir()

    # --- construcción de la app ---------------------------------------- #
    def _construir(self):
        from prompt_toolkit.application import Application
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.document import Document
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
        from prompt_toolkit.key_binding.defaults import load_key_bindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import (
            ConditionalContainer, Float, FloatContainer, HSplit, VSplit, Window,
            WindowAlign)
        from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
        from prompt_toolkit.layout.dimension import Dimension
        from prompt_toolkit.layout.menus import CompletionsMenu
        from prompt_toolkit.layout.processors import BeforeInput
        from prompt_toolkit.lexers import Lexer
        from prompt_toolkit.styles import Style

        self._Document = Document
        tui = self

        class _ComandoCompleter(Completer):
            def get_completions(self, document, complete_event):
                texto = document.text_before_cursor
                if texto.startswith("/") and " " not in texto:
                    for cmd in CHAT_COMMANDS:
                        for nombre in cmd["names"]:
                            if nombre.startswith(texto.lower()):
                                yield Completion(
                                    nombre, start_position=-len(texto),
                                    display=nombre, display_meta=cmd["help"],
                                    style=f"fg:{COMANDO_AZUL}")
                                break
                    return
                partes = texto.split(" ", 1)
                if len(partes) == 2 and partes[0].lower() == "/color":
                    arg = partes[1]
                    for opcion in COLOR_OPCIONES:
                        if opcion.startswith(arg.strip().lower()):
                            yield Completion(opcion, start_position=-len(arg),
                                             display=opcion,
                                             display_meta=f"barras en {opcion}")

        class _ComandoLexer(Lexer):
            def lex_document(self, document):
                def obtener(numero):
                    linea = document.lines[numero]
                    if numero == 0 and linea.startswith("/"):
                        corte = linea.find(" ")
                        if corte == -1:
                            return [(f"fg:{COMANDO_AZUL} bold", linea)]
                        return [(f"fg:{COMANDO_AZUL} bold", linea[:corte]),
                                ("", linea[corte:])]
                    return [("", linea)]
                return obtener

        # Ventana de salida (la conversación; hace scroll hacia arriba).
        self.salida = Buffer(document=Document("", 0))
        salida_win = Window(
            BufferControl(buffer=self.salida, focusable=False),
            wrap_lines=True)

        # Ventana de candidatas (solo motor llamaserver). Se mantiene visible tras
        # generar para poder leer las candidatas del último token (antes se ocultaba
        # al terminar y pasaban volando). Se limpia al empezar la siguiente generación.
        cand_win = ConditionalContainer(
            Window(FormattedTextControl(lambda: self._candidatas_ft()),
                   dont_extend_height=True, height=Dimension(min=1), wrap_lines=True),
            filter=Condition(lambda: self.muestra_candidatas and self.candidatas_vivo
                             and bool(self.candidatas["lista"])))

        def _color():
            return f"fg:{self.tema['color']}"

        barra_sup = Window(char="─", height=1, style=_color)
        barra_inf = Window(char="─", height=1, style=_color)

        self.entrada = Buffer(completer=_ComandoCompleter(),
                              complete_while_typing=True, multiline=False,
                              history=InMemoryHistory(),
                              accept_handler=self._on_accept)
        entrada_win = Window(
            BufferControl(buffer=self.entrada, lexer=_ComandoLexer(),
                          input_processors=[BeforeInput("› ", style="class:indicador")]),
            wrap_lines=True, dont_extend_height=True, height=Dimension(min=1))

        fila_estado = VSplit([
            Window(FormattedTextControl(lambda: self._status_izq()), height=1,
                   style="class:estado"),
            Window(FormattedTextControl(lambda: self._status_der()), height=1,
                   style="class:estado", align=WindowAlign.RIGHT,
                   dont_extend_width=True),
        ], height=1)

        cuerpo = HSplit([salida_win, cand_win, barra_sup, entrada_win,
                         barra_inf, fila_estado])
        raiz = FloatContainer(content=cuerpo, floats=[
            Float(xcursor=True, ycursor=True,
                  content=CompletionsMenu(max_height=10, scroll_offset=1)),
        ])

        kb = KeyBindings()

        @kb.add("c-c")
        def _(event):
            # Generando: 1ª pulsación cancela (recupera el control); 2ª fuerza salir.
            # Sin generar: sale de la aplicación.
            if self.generando and not self._cancelar.is_set():
                self._solicitar_cancelar()
            else:
                event.app.exit()

        @kb.add("escape")
        def _(event):
            if self.generando:  # Esc también cancela la generación en curso
                self._solicitar_cancelar()

        @kb.add("c-d")
        def _(event):
            if not self.entrada.text and not self.generando:
                event.app.exit()

        # Desplazar la salida con el teclado (sin ratón: evita basura por mouse).
        @kb.add("pageup")
        def _(event):
            self.salida.cursor_up(12)

        @kb.add("pagedown")
        def _(event):
            self.salida.cursor_down(12)

        estilo = Style.from_dict({
            "estado": "fg:#9e9e9e",
            "indicador": f"bold fg:{COMANDO_AZUL}",
            "completion-menu.completion": f"bg:#1c1c1c fg:{COMANDO_AZUL}",
            "completion-menu.completion.current": f"bg:{COMANDO_AZUL} fg:#ffffff",
            "completion-menu.meta.completion": "bg:#1c1c1c fg:#808080",
            "completion-menu.meta.completion.current": f"bg:{COMANDO_AZUL} fg:#e4e4e4",
        })

        self.app = Application(
            layout=Layout(raiz, focused_element=entrada_win),
            key_bindings=merge_key_bindings([load_key_bindings(), kb]),
            # Sin ratón: en algunos terminales el movimiento del ratón filtraba
            # secuencias de escape al prompt. El scroll se hace con RePág/AvPág.
            style=estilo, full_screen=True, mouse_support=False)

    # --- render ---------------------------------------------------------- #
    def _texto_salida(self) -> str:
        partes = list(self.transcripcion)
        if self.generando or self.stream_text:
            partes.append("LlamaVino: " + self.stream_text)
        return "\n\n".join(p for p in partes if p)

    def _refrescar(self):
        texto = self._texto_salida()
        # Cursor al final => la ventana se desplaza al fondo (sigue la salida).
        self.salida.set_document(self._Document(texto, len(texto)),
                                 bypass_readonly=True)

    def _candidatas_ft(self):
        estado = "" if self.generando else "  (último token)"
        lineas = [("bold cyan",
                   f"┌─ Candidatas del próximo token (n_probs){estado} ──\n")]
        for cand in self.candidatas["lista"]:
            elegida = cand["word"] == self.candidatas["elegida"]
            estilo = "bold green" if elegida else "fg:#808080"
            flecha = "│ → " if elegida else "│   "
            lineas.append((estilo,
                           f"{flecha}{cand['word']!r}  {cand['prob'] * 100:5.1f}%\n"))
        return lineas

    def _status_izq(self) -> str:
        # Siempre empieza fijo por «LlamaVINO |»; luego ruta, modelo y contexto.
        if self.ctx_len:
            pct = min(100, round(self.tokens_vivos / self.ctx_len * 100))
            ctx = f"Contexto: {pct}% usado"
        else:
            ctx = "Contexto: n/d"
        partes = ["LlamaVINO", self.ruta_dir, self.nombre_modelo, ctx]
        # Token candidato elegido en tiempo real, entre el contexto y los tokens
        # (solo con llama-server y si la vista en vivo está activada).
        if self.muestra_candidatas and self.candidatas_vivo and self.candidatas["elegida"]:
            partes.append(f"cand: {self.candidatas['elegida']!r}")
        return " " + " | ".join(partes)

    def _status_der(self) -> str:
        # «i / j tokens»: i = tokens en uso en vivo, j = máximo de la sesión (contexto).
        if self.ctx_len:
            return f"{self.tokens_vivos} / {self.ctx_len} tokens "
        return f"{self.tokens_vivos} tokens "

    def _log(self, texto: str):
        self.transcripcion.append(texto)
        self._refrescar()
        self.app.invalidate()

    def _capturar_texto(self, func) -> str:
        """Renderiza ``func(console)`` (rich) a texto plano y lo devuelve."""
        import io
        import shutil
        from rich.console import Console as _RC

        ancho = max(40, shutil.get_terminal_size((100, 24)).columns)
        buf = io.StringIO()
        func(_RC(file=buf, width=ancho))
        return buf.getvalue().rstrip("\n")

    async def _pager(self, func):
        """Muestra salida larga (p. ej. /ir, /gguf) en el terminal con pausa.

        Sale temporalmente de la pantalla completa (``run_in_terminal``) para
        imprimir en el terminal normal —donde el scroll del ratón funciona— y
        espera a ENTER, de modo que dé tiempo a leerlo. Luego vuelve al chat.
        """
        from prompt_toolkit.application import run_in_terminal

        salida = self._capturar_texto(func)

        def _mostrar():
            print(salida)
            try:
                input("\n— Pulsa ENTER para volver al chat (desplázate para leer) —")
            except EOFError:
                pass

        await run_in_terminal(_mostrar)

    # --- entrada / comandos --------------------------------------------- #
    def _on_accept(self, buff) -> bool:
        if self.generando:
            return True  # ignora Enter mientras genera (conserva el texto)
        texto = buff.text.strip()
        if texto:
            self.app.create_background_task(self._procesar(texto))
        return False  # limpia el input

    async def _procesar(self, texto: str):
        # Captura cualquier fallo para no romper el bucle de eventos de la TUI.
        try:
            if texto.startswith("/"):
                await self._comando(texto)
            else:
                await self._generar(texto)
        except Exception as exc:  # noqa: BLE001 - se informa en el chat, sin crash
            self.generando = False
            self._log(f"[error] {type(exc).__name__}: {exc}")

    async def _comando(self, texto: str):
        from prompt_toolkit.application import run_in_terminal

        cmd = _resolve_command(texto)
        if cmd is None:
            self._log(f"Comando desconocido: {texto.split()[0]} — escribe /help")
            return
        if cmd == "/exit":
            self.app.exit()
        elif cmd == "/model":
            # Sale a elegir modelo con flechas y reentra conservando la conversación.
            self.accion = "switch"
            self.app.exit()
        elif cmd == "/help":
            await self._pager(_print_help)
        elif cmd == "/status":
            await self._pager(lambda c: _print_status(c, self.model_path, self.engine,
                                                      self.settings))
        elif cmd == "/cost":
            await self._pager(lambda c: _print_cost(c, self.stats))
        elif cmd == "/doctor":
            await self._pager(_run_doctor)
        elif cmd == "/mcp":
            await self._pager(lambda c: _list_mcp_servers(c, self.workspace))
        elif cmd == "/candidatas":
            await self._cmd_candidatas(texto)
        elif cmd == "/gpu":
            await self._pager(lambda c: _show_gpu_plan(c, self.engine, self.model_path,
                                                       self.device, self.args))
        elif cmd == "/gguf":
            resto = texto.split(maxsplit=1)
            filtro, limite = _parsear_args_gguf(resto[1] if len(resto) > 1 else "")
            await self._pager(lambda c: _show_gguf(c, self.model_path, filtro, limite))
        elif cmd == "/ir":
            await self._pager(lambda c: _show_ir(c, self.model_path))
        elif cmd == "/color":
            resto = texto.split(maxsplit=1)
            self._log(self._capturar_texto(
                lambda c: _cmd_color(c, self.tema,
                                     resto[1].strip() if len(resto) > 1 else "")))
            guardar_ajustes(self.settings, self.tema["color"])
            self.app.invalidate()
        elif cmd == "/clear":
            self.turns = []
            self.tokens_vivos = _tokens_contexto(self.turns)
            self._log("Conversación reiniciada.")
        elif cmd == "/config":
            # Sale al editor de parámetros (con flechas) y reentra.
            self.accion = "config"
            self.app.exit()
        elif cmd == "/save":
            resto = texto.split(maxsplit=1)
            explicit = resto[1].strip() if len(resto) > 1 else None
            if not self.last_response:
                self._log("Aún no hay ninguna respuesta que guardar.")
            else:
                await run_in_terminal(lambda: _save_from_response(
                    self.console, self.last_response, self.workspace, explicit))
        elif cmd == "/compact":
            await self._compactar()

    # --- generación en segundo plano ------------------------------------ #
    def _streamer(self, piece: str):
        self.stream_text += piece
        self._n_tok += 1
        self.tokens_vivos = self._ctx_base + self._n_tok
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._tick)

    def _tick(self):
        self._refrescar()
        self.app.invalidate()

    def _on_cands(self, cands, elegida):
        self.candidatas = {"lista": cands, "elegida": elegida}
        # Registra siempre el histórico (aunque la vista en vivo esté apagada).
        self.hist_candidatas.append({"pos": len(self.hist_candidatas) + 1,
                                     "elegida": elegida, "lista": list(cands)})
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.app.invalidate)

    def _trabajo_generar(self, turns, settings):
        return _motor_generar(self.engine, turns, settings, self.ov,
                              self._streamer, on_candidates=self._on_cands,
                              debe_parar=self._cancelar.is_set)

    def _solicitar_cancelar(self):
        """Cancela la generación en curso de forma suave (recupera el control).

        Activa el evento que detiene el streamer de OpenVINO y pide al motor de
        respaldo (llama.cpp / llama-server) que aborte su subproceso/petición.
        """
        if not self.generando:
            return
        self._cancelar.set()
        obj = self.engine.get("obj")
        cancelar = getattr(obj, "cancelar", None)
        if callable(cancelar):
            try:
                cancelar()
            except Exception:  # noqa: BLE001 - mejor esfuerzo
                pass
        self._log("(cancelando… pulsa Ctrl-C de nuevo para salir)")
        self.app.invalidate()

    async def _cmd_candidatas(self, texto: str):
        """/candidatas [on|off|historico]: alterna la vista en vivo o revisa el histórico."""
        if not self.muestra_candidatas:
            self._log("Las candidatas solo están disponibles con --engine llamaserver "
                      "(p. ej. python LlamaVino.py --engine llamaserver -m modelo.gguf).")
            return
        resto = texto.split(maxsplit=1)
        arg = resto[1].strip().lower() if len(resto) > 1 else ""
        if arg in ("historico", "histórico", "hist", "historial"):
            if not self.hist_candidatas:
                self._log("Aún no hay histórico de candidatas (genera una respuesta antes).")
                return
            hist = list(self.hist_candidatas)
            await self._pager(lambda c: _print_candidatas_hist(c, hist))
            return
        if arg == "on":
            self.candidatas_vivo = True
        elif arg == "off":
            self.candidatas_vivo = False
        elif arg == "":
            self.candidatas_vivo = not self.candidatas_vivo  # alterna
        else:
            self._log("Uso: /candidatas [on|off|historico]")
            return
        guardar_ajustes(candidatas_vivo=self.candidatas_vivo)
        estado = "ACTIVADA" if self.candidatas_vivo else "DESACTIVADA"
        self._log(f"Vista en vivo de candidatas {estado} "
                  "(el histórico se sigue registrando: /candidatas historico).")
        self.app.invalidate()

    async def _generar(self, texto: str):
        import asyncio

        from prompt_toolkit.application import run_in_terminal

        self.turns.append({"role": "user", "content": texto})
        self.transcripcion.append(f"[Tú] {texto}")
        self.stream_text = ""
        self.candidatas = {"lista": [], "elegida": ""}
        self.hist_candidatas = []  # nuevo histórico para esta generación
        self._ctx_base = _tokens_contexto(self.turns)
        self._n_tok = 0
        self._inicio = time.perf_counter()
        self._cancelar.clear()
        self.generando = True
        self._loop = asyncio.get_event_loop()
        self._refrescar()
        self.app.invalidate()

        try:
            resultado, _ = await self._loop.run_in_executor(
                None, self._trabajo_generar, self.turns, self.settings)
        except Exception as exc:  # noqa: BLE001 - cualquier fallo del motor
            self.generando = False
            self.stream_text = ""
            self.turns.pop()  # descarta el turno de usuario sin respuesta
            self._log(f"[error del motor] {type(exc).__name__}: {exc}")
            return
        finally:
            self.generando = False

        if self._cancelar.is_set():
            # Generación cancelada: conserva lo ya producido como respuesta parcial.
            parcial = self.stream_text.strip()
            self.stream_text = ""
            if parcial:
                self.last_response = parcial
                self.turns.append({"role": "assistant", "content": parcial})
                self.transcripcion.append("LlamaVino: " + parcial)
            else:
                self.turns.pop()  # nada generado: descarta el turno de usuario
            self.transcripcion.append("(generación cancelada)")
            self.candidatas = {"lista": [], "elegida": ""}
            self.tokens_vivos = _tokens_contexto(self.turns)
            self._refrescar()
            self.app.invalidate()
            return

        elapsed = time.perf_counter() - self._inicio
        self.last_response = str(resultado)
        self.turns.append({"role": "assistant", "content": self.last_response})
        self.stats["turns"] += 1
        self.stats["output_tokens"] += self._n_tok
        self.stats["seconds"] += elapsed
        # Registra la velocidad (tok/s) en la BD de rendimiento.
        if elapsed > 0 and self._n_tok > 0 and self.model_path:
            try:
                perf_db.registrar_generacion(Path(self.model_path).name,
                                             self._n_tok / elapsed)
            except Exception:  # noqa: BLE001 - best-effort
                pass
        self.stream_text = ""
        self.transcripcion.append("LlamaVino: " + self.last_response)
        self.transcripcion.append(f"({elapsed:.1f}s · {self._n_tok} tok · "
                                  f"{self._n_tok / elapsed:.1f} tok/s)")
        self.tokens_vivos = _tokens_contexto(self.turns)
        self._refrescar()
        self.app.invalidate()

        # Autoguarda los bloques con nombre (necesita la terminal para confirmar).
        bloques = [b for b in extract_code_blocks(self.last_response) if b["filename"]]
        if bloques:
            def _guardar():
                for bloque in bloques:
                    _confirm_and_write(self.console, bloque["filename"],
                                       bloque["code"], self.workspace)
            await run_in_terminal(_guardar)


    async def _compactar(self):
        import asyncio

        if not self.turns:
            self._log("No hay conversación que compactar.")
            return
        transcript = "\n".join(f"{t['role']}: {t['content']}" for t in self.turns)
        prompt = ("Resume en español, de forma concisa, la siguiente conversación. "
                  "Conserva los hechos, decisiones y contexto importantes:\n\n"
                  + transcript)
        ajustes = {"max_new_tokens": 512, "temperature": 0.0, "top_p": 0.9,
                   "top_k": 40, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                   "stop_strings": []}
        self.stream_text = ""
        self._ctx_base = _tokens_contexto(self.turns)
        self._n_tok = 0
        self._inicio = time.perf_counter()
        self._cancelar.clear()
        self.generando = True
        self._loop = asyncio.get_event_loop()
        self._log("Compactando conversación…")
        try:
            resumen, _ = await self._loop.run_in_executor(
                None, self._trabajo_generar,
                [{"role": "user", "content": prompt}], ajustes)
        except Exception as exc:  # noqa: BLE001 - cualquier fallo del motor
            self.generando = False
            self.stream_text = ""
            self._log(f"[error del motor] {type(exc).__name__}: {exc}")
            return
        finally:
            self.generando = False
        self.stream_text = ""
        self.turns = [{"role": "system",
                       "content": "Resumen de la conversación previa:\n" + str(resumen)}]
        self.tokens_vivos = _tokens_contexto(self.turns)
        self._log("Conversación compactada.")

    def exportar_estado(self) -> dict:
        """Devuelve el estado a conservar al reentrar (tras /model o /config)."""
        return {
            "settings": self.settings, "turns": self.turns, "stats": self.stats,
            "last_response": self.last_response, "tema": self.tema,
            "candidatas_vivo": self.candidatas_vivo,
        }

    def run(self) -> str | None:
        self.tokens_vivos = _tokens_contexto(self.turns)
        if not self.transcripcion:
            self.transcripcion.append(
                "Chat listo. Escribe «/» para ver los comandos. La salida hace "
                "scroll arriba; las barras, el prompt y el estado quedan fijos abajo.")
        self._refrescar()
        try:
            self.app.run()
        except KeyboardInterrupt:
            # En Windows, un Ctrl-C que llega tras devolver el control (durante el
            # cierre del bucle asyncio) reaparece como señal real. La app ya salió;
            # lo absorbemos para no ensuciar con un traceback.
            pass
        return self.accion


def interactive_session(args) -> int:
    """Bucle interactivo principal: elegir modelo (con flechas), chatear, cambiar.

    Cada vez que dentro del chat se pide ``/model`` o ``/config``, la TUI sale, se
    ejecuta el selector/editor con flechas y se reentra **conservando la
    conversación** (sin reiniciar el proceso ni recargar el modelo en /config).
    """
    import openvino_genai as ov_genai
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print(f"[bold cyan]{BANNER}[/]")
    console.print(f"[dim]{AUTOR} · v{__version__}[/]")
    console.print(
        Panel.fit(
            "GGUF/IR + OpenVINO en Intel Iris Xe (con respaldo llama.cpp).\n"
            "Elige un modelo con ↑/↓ y Enter; se descarga si hace falta.",
            border_style="cyan",
        )
    )

    # Selección inicial del modelo.
    model_path = interactive_select_model(console, args)
    if model_path is None:
        console.print("[dim]Hasta luego.[/]")
        return 0
    device = pick_device(args.device)
    try:
        engine = crear_motor(model_path, args, device)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]No se pudo cargar el modelo: {exc}[/]")
        return 1

    estado = None
    while True:
        tui = ChatTUI(console, engine, args, model_path, device, ov_genai,
                      estado=estado)
        accion = tui.run()
        estado = tui.exportar_estado()  # conserva conversación/ajustes/color

        if accion == "switch":
            nuevo = interactive_select_model(console, args)
            if nuevo is not None:
                cerrar_motor(engine)
                model_path = nuevo
                device = pick_device(args.device)
                try:
                    engine = crear_motor(model_path, args, device)
                except Exception as exc:  # noqa: BLE001
                    console.print(f"[red]No se pudo cargar el modelo: {exc}[/]")
                    return 1
            continue
        if accion == "config":
            configure_settings(console, estado["settings"])  # editor con flechas
            guardar_ajustes(estado["settings"], estado["tema"]["color"])
            continue
        # Salida normal del chat -> terminar.
        cerrar_motor(engine)
        return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="LlamaVino",
        description="Ejecuta un modelo GGUF/IR con OpenVINO (o llama.cpp) en una "
        "GPU Intel Iris Xe. © Rafael Ausejo Prieto, con ayuda de Claude Code.",
    )
    parser.add_argument(
        "--version", action="version", version=f"LlamaVino {__version__}",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Lista los dispositivos OpenVINO disponibles y sale.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Lista los modelos GGUF recomendados y sale.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Lanza la interfaz interactiva (selector de modelo + chat). Es "
        "también lo predeterminado si no se da --model/--prompt/--download.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Ejecuta el backend de líneas JSON en stdin/stdout (usado por la UI Ink).",
    )
    parser.add_argument(
        "--download",
        metavar="ALIAS|REPO:FICHERO",
        help="Descarga un modelo GGUF (alias del registro o "
        "'repo_id:fichero.gguf') en --models-dir; genera después si se da --prompt.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directorio donde se guardan y buscan los modelos (por defecto: models).",
    )
    parser.add_argument(
        "--convert-ir",
        metavar="HF_ID",
        help="Convierte un modelo de Hugging Face a OpenVINO IR (optimum-cli) en "
        "--models-dir y sale. Carga el directorio resultante con -m. Requiere "
        "optimum-intel[openvino].",
    )
    parser.add_argument(
        "--weight-format",
        choices=["int4", "int8", "fp16"],
        default="int4",
        help="Cuantización de pesos al convertir a IR (--convert-ir). Por defecto: int4.",
    )
    parser.add_argument(
        "-m", "--model", type=Path, help="Ruta al fichero de modelo .gguf cuantizado."
    )
    parser.add_argument(
        "-p",
        "--prompt",
        help="Prompt a generar. Si se omite, se lee de stdin.",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="AUTO",
        help="Dispositivo OpenVINO: AUTO (prefiere GPU), GPU, CPU, ... (por defecto: AUTO).",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "openvino", "llamacpp", "llamaserver"],
        default="auto",
        help="Motor: openvino (carga todo en el dispositivo), llamacpp (mmap + "
        "reparto de capas GPU/CPU), llamaserver (igual + panel de palabras "
        "candidatas por token vía llama-server) o auto. Por defecto: auto.",
    )
    parser.add_argument(
        "--n-probs",
        type=int,
        default=5,
        help="Nº de palabras candidatas a mostrar por token (motor llamaserver).",
    )
    parser.add_argument(
        "--n-gpu-layers",
        default="auto",
        help="Capas a la GPU para el motor llama.cpp: 'auto' (calcula el reparto "
        "óptimo en tiempo real) o un entero. Por defecto: auto.",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=4096,
        help="Tamaño de contexto para el motor llama.cpp (por defecto: 4096).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Número máximo de tokens a generar (por defecto: 512).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperatura de muestreo; 0 desactiva el muestreo (greedy). Por defecto: 0.0.",
    )
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Muestreo por núcleo (nucleus) top-p.")
    parser.add_argument("--top-k", type=int, default=40, help="Corte de muestreo top-k.")
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="Penalización de presencia (-2.0..2.0); castiga a los tokens por aparecer.",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.0,
        help="Penalización de frecuencia (-2.0..2.0); castiga según cuánto se repiten.",
    )
    parser.add_argument(
        "--stop",
        action="append",
        metavar="SEC",
        help="Secuencia de parada; la generación se detiene al emitirla. Repetible.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Desactiva el streaming de tokens; imprime el resultado completo al final.",
    )
    return parser.parse_args(argv)


def _validar_args(args) -> str | None:
    """Valida rangos de los flags. Devuelve un mensaje de error o None si todo OK."""
    if not (0.0 <= args.temperature <= 2.0):
        return "--temperature debe estar entre 0.0 y 2.0."
    if not (0.0 <= args.top_p <= 1.0):
        return "--top-p debe estar entre 0.0 y 1.0."
    if args.top_k < 0:
        return "--top-k no puede ser negativo."
    if args.max_new_tokens < 1:
        return "--max-new-tokens debe ser >= 1."
    if args.n_ctx < 1:
        return "--n-ctx debe ser >= 1."
    if args.n_probs < 1:
        return "--n-probs debe ser >= 1."
    for nombre, valor in (("--presence-penalty", args.presence_penalty),
                          ("--frequency-penalty", args.frequency_penalty)):
        if not (-2.0 <= valor <= 2.0):
            return f"{nombre} debe estar entre -2.0 y 2.0."
    ngl = str(getattr(args, "n_gpu_layers", "auto")).lower()
    if ngl != "auto" and not ngl.isdigit():
        return "--n-gpu-layers debe ser 'auto' o un entero >= 0."
    return None


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    args = parse_args(argv)

    error = _validar_args(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    if args.serve:
        return serve_stdio()

    if args.list_devices:
        describe_devices()
        return 0

    if args.list_models:
        list_models()
        return 0

    if args.convert_ir:
        salida = convert_to_ir(args.convert_ir, args.models_dir, args.weight_format)
        print(f'Cárgalo con: python LlamaVino.py -m "{salida}"')
        return 0

    # Interfaz interactiva: flag explícito, o sin argumentos accionables en una terminal.
    wants_interactive = args.interactive or (
        args.model is None
        and args.prompt is None
        and args.download is None
        and sys.stdin.isatty()
    )
    if wants_interactive:
        return interactive_session(args)

    if args.download:
        downloaded = download_model(args.download, args.models_dir)
        # Usa el fichero recién descargado salvo que se indicara un --model explícito.
        if args.model is None:
            args.model = downloaded
        # Sólo descarga: sin un prompt no hay nada más que hacer.
        if args.prompt is None and sys.stdin.isatty():
            print("Descarga completada. Añade --prompt para generar también.")
            return 0

    if args.model is None:
        print("Error: --model es obligatorio (ruta a un .gguf o a un directorio IR).",
              file=sys.stderr)
        return 2
    if not (args.model.is_file() or es_modelo_ir(args.model)):
        print(f"Error: no se encontró el modelo (.gguf o directorio IR): {args.model}",
              file=sys.stderr)
        return 2

    prompt = args.prompt
    if prompt is None:
        if sys.stdin.isatty():
            print("Error: no se dio ningún prompt. Usa --prompt o canaliza texto por stdin.",
                  file=sys.stderr)
            return 2
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("Error: prompt vacío.", file=sys.stderr)
        return 2

    device = pick_device(args.device)
    import openvino_genai as ov_genai

    engine = crear_motor(args.model, args, device)

    def streamer(subword: str) -> None:
        if not args.no_stream:
            print(subword, end="", flush=True)

    settings = default_gen_settings(args)
    print("\n--- generación ---")
    start = time.perf_counter()
    try:
        result, _ = _motor_generar(
            engine, [{"role": "user", "content": prompt}], settings, ov_genai, streamer)
    finally:
        cerrar_motor(engine)
    elapsed = time.perf_counter() - start

    if args.no_stream:
        print(result)
    print(f"\n--- hecho en {elapsed:.1f}s ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
