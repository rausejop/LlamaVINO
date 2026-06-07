#!/usr/bin/env python
"""Lector de información de un modelo OpenVINO IR (sin dependencias externas).

Un modelo IR es una carpeta con ``openvino_model.xml`` + ``openvino_model.bin``
(los pesos) y ficheros de configuración/tokenizador. Este módulo reúne, sin
cargar el modelo, la información útil del «fichero binario»: tamaños de los
ficheros, la configuración del modelo (``config.json``), la de generación
(``generation_config.json``) y el ``rt_info`` del XML (versión de optimum,
compresión de pesos, etc.), con explicaciones en español. Es el equivalente a
``gguf_reader`` pero para el formato IR.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def _parse_xml_seguro(texto: str):
    """Parsea XML mitigando XXE/billion-laughs.

    Usa ``defusedxml`` si está instalado; si no, rechaza cualquier fragmento con
    ``<!DOCTYPE``/``<!ENTITY`` (vectores de esos ataques) antes de recurrir al
    parser de la biblioteca estándar. Devuelve el elemento raíz o ``None``.
    """
    if "<!DOCTYPE" in texto or "<!ENTITY" in texto:
        return None
    try:
        import defusedxml.ElementTree as _DET
        return _DET.fromstring(texto)
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - XML inválido/peligroso
        return None
    try:
        return ET.fromstring(texto)
    except ET.ParseError:
        return None

# Significado en español de las claves de ``config.json`` más relevantes.
CONFIG_MEANINGS: dict[str, str] = {
    "model_type": "Tipo de modelo (llama, qwen2, phi3, ...).",
    "architectures": "Clase de arquitectura de Transformers.",
    "hidden_size": "Dimensión del estado oculto (embedding).",
    "num_hidden_layers": "Número de capas transformer.",
    "num_attention_heads": "Número de cabezas de atención.",
    "num_key_value_heads": "Cabezas clave/valor (GQA si es menor que las de atención).",
    "head_dim": "Dimensión por cabeza de atención.",
    "intermediate_size": "Dimensión interna de la capa feed-forward.",
    "vocab_size": "Tamaño del vocabulario.",
    "max_position_embeddings": "Longitud máxima de contexto en tokens.",
    "rope_theta": "Frecuencia base de RoPE (codificación posicional rotatoria).",
    "rms_norm_eps": "Épsilon del RMSNorm (estabilidad numérica).",
    "sliding_window": "Ventana de atención deslizante (si aplica).",
    "torch_dtype": "Tipo de dato original de los pesos.",
    "tie_word_embeddings": "Comparte pesos de embedding de entrada y salida.",
    "bos_token_id": "ID del token de inicio de secuencia (BOS).",
    "eos_token_id": "ID del token de fin de secuencia (EOS).",
    "pad_token_id": "ID del token de relleno (padding).",
}


def _leer_json(path: Path) -> dict:
    """Lee un JSON y devuelve un dict (vacío si no existe o es inválido)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _leer_rt_info(xml_path: Path) -> list[dict]:
    """Extrae el bloque ``<rt_info>`` del XML del IR (lee solo la cola del fichero).

    El ``rt_info`` está al final del ``.xml``, así que basta leer los últimos
    kilobytes en vez de parsear todas las capas. Devuelve ``[{"key", "value"}]``.
    """
    try:
        tam = xml_path.stat().st_size
        with open(xml_path, "rb") as handle:
            if tam > 65536:
                handle.seek(-65536, 2)
            cola = handle.read().decode("utf-8", "replace")
    except OSError:
        return []
    match = re.search(r"<rt_info>.*</rt_info>", cola, re.DOTALL)
    if not match:
        return []
    raiz = _parse_xml_seguro(match.group(0))
    if raiz is None:
        return []
    salida: list[dict] = []

    def recorrer(nodo, prefijo: str) -> None:
        valor = nodo.get("value")
        if valor is not None:
            salida.append({"key": prefijo, "value": valor})
        for hijo in nodo:
            recorrer(hijo, f"{prefijo}/{hijo.tag}")

    for hijo in raiz:
        recorrer(hijo, hijo.tag)
    return salida


def _formatear(valor) -> str:
    """Convierte un valor de config en una cadena legible y acotada."""
    if isinstance(valor, list):
        return ", ".join(str(v) for v in valor) if valor else "[]"
    if isinstance(valor, dict):
        return "{…}"
    texto = str(valor)
    return texto if len(texto) <= 80 else texto[:77] + "..."


def describe_ir(path: str | Path) -> dict:
    """Reúne la información de un modelo IR para mostrarla.

    Returns:
      ``{"name", "total_bytes", "architecture", "files": [{"name","bytes"}],
         "config_rows": [{"key","value","meaning"}], "gen_rows": [{"key","value"}],
         "rt_info": [{"key","value"}]}``.

    Raises:
      ValueError: Si ``path`` no es un directorio de modelo IR.
    """
    base = Path(path)
    if not (base.is_dir() and (base / "openvino_model.xml").is_file()):
        raise ValueError(f"no es un modelo OpenVINO IR: {path}")

    ficheros = sorted((f.name, f.stat().st_size) for f in base.iterdir() if f.is_file())
    total = sum(tam for _, tam in ficheros)

    config = _leer_json(base / "config.json")
    architecture = config.get("model_type")
    if not architecture and isinstance(config.get("architectures"), list):
        architecture = (config["architectures"] or [None])[0]

    config_rows = [
        {"key": clave, "value": _formatear(valor),
         "meaning": CONFIG_MEANINGS.get(clave, "")}
        for clave, valor in config.items()
    ]
    gen = _leer_json(base / "generation_config.json")
    gen_rows = [{"key": k, "value": _formatear(v)} for k, v in gen.items()
                if not isinstance(v, (dict, list))]
    rt_info = _leer_rt_info(base / "openvino_model.xml")

    return {
        "name": base.name,
        "total_bytes": total,
        "architecture": architecture,
        "files": [{"name": n, "bytes": s} for n, s in ficheros],
        "config_rows": config_rows,
        "gen_rows": gen_rows,
        "rt_info": rt_info,
    }
