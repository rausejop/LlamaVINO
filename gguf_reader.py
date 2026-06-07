#!/usr/bin/env python
"""Lector de la cabecera de un fichero GGUF (sin dependencias externas).

GGUF guarda, antes de los tensores, un bloque de metadatos clave-valor con la
configuración del modelo (arquitectura, longitud de contexto, tokenizador, ...).
Este módulo lee SÓLO ese bloque de metadatos y aporta una explicación en español
del significado de cada clave conocida. No depende del modelo ni de OpenVINO.

Formato (todo little-endian):
    magic   : 4 bytes  == b"GGUF"
    version : uint32
    n_tensors : uint64
    n_kv      : uint64
    n_kv veces: <string clave><uint32 tipo><valor>

Referencia: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from pathlib import Path

GGUF_MAGIC = b"GGUF"

# Identificadores de tipo de valor del formato GGUF.
(
    _UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _BOOL,
    _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64,
) = range(13)

# Tipo escalar -> (formato struct, tamaño en bytes).
_SCALAR_FMT: dict[int, tuple[str, int]] = {
    _UINT8: ("<B", 1), _INT8: ("<b", 1),
    _UINT16: ("<H", 2), _INT16: ("<h", 2),
    _UINT32: ("<I", 4), _INT32: ("<i", 4),
    _FLOAT32: ("<f", 4), _BOOL: ("<?", 1),
    _UINT64: ("<Q", 8), _INT64: ("<q", 8),
    _FLOAT64: ("<d", 8),
}

# Valor de ``general.file_type`` (enum LLAMA_FTYPE de llama.cpp) -> nombre.
FILE_TYPE_NAMES: dict[int, str] = {
    0: "ALL_F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0",
    9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS",
    24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
    29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
}

# Significado en español de las claves de metadatos. ``{arch}`` se sustituye por
# la arquitectura real del modelo (p. ej. ``llama``, ``qwen2``).
KEY_MEANINGS: dict[str, str] = {
    "general.architecture": "Arquitectura del modelo (llama, qwen2, ...).",
    "general.name": "Nombre del modelo.",
    "general.basename": "Nombre base del modelo.",
    "general.finetune": "Variante de ajuste fino (p. ej. Instruct).",
    "general.size_label": "Etiqueta de tamaño (p. ej. 3B).",
    "general.license": "Licencia del modelo.",
    "general.quantization_version": "Versión del esquema de cuantización.",
    "general.file_type": "Tipo de cuantización del fichero (p. ej. Q4_K_M).",
    "{arch}.context_length": "Longitud máxima de contexto en tokens.",
    "{arch}.embedding_length": "Dimensión del embedding (estado oculto).",
    "{arch}.block_count": "Número de capas/bloques transformer.",
    "{arch}.feed_forward_length": "Dimensión interna de la capa feed-forward.",
    "{arch}.attention.head_count": "Número de cabezas de atención.",
    "{arch}.attention.head_count_kv":
        "Cabezas de clave/valor (GQA si es menor que head_count).",
    "{arch}.attention.layer_norm_rms_epsilon":
        "Épsilon del RMSNorm (estabilidad numérica).",
    "{arch}.attention.layer_norm_epsilon": "Épsilon de la normalización de capa.",
    "{arch}.rope.freq_base":
        "Frecuencia base de RoPE (codificación posicional rotatoria).",
    "{arch}.rope.dimension_count": "Dimensiones a las que se aplica RoPE.",
    "{arch}.rope.scaling.type": "Tipo de escalado de RoPE (para extender contexto).",
    "{arch}.vocab_size": "Tamaño del vocabulario.",
    "{arch}.expert_count": "Número de expertos (modelos MoE).",
    "{arch}.expert_used_count": "Expertos activos por token (MoE).",
    "tokenizer.ggml.model": "Tipo de tokenizador (bpe, spm, ...).",
    "tokenizer.ggml.pre": "Variante de pre-tokenización.",
    "tokenizer.ggml.tokens": "Lista de tokens del vocabulario.",
    "tokenizer.ggml.scores": "Puntuación de cada token.",
    "tokenizer.ggml.token_type": "Tipo de cada token (normal, control, ...).",
    "tokenizer.ggml.merges": "Reglas de fusión BPE.",
    "tokenizer.ggml.bos_token_id": "ID del token de inicio de secuencia (BOS).",
    "tokenizer.ggml.eos_token_id": "ID del token de fin de secuencia (EOS).",
    "tokenizer.ggml.padding_token_id": "ID del token de relleno (padding).",
    "tokenizer.ggml.unknown_token_id": "ID del token desconocido (UNK).",
    "tokenizer.ggml.add_bos_token": "Si se añade BOS automáticamente.",
    "tokenizer.ggml.add_eos_token": "Si se añade EOS automáticamente.",
    "tokenizer.chat_template": "Plantilla de chat (formato de los mensajes).",
}


def _read_exact(handle, size: int) -> bytes:
    """Lee ``size`` bytes o lanza ``ValueError`` si el fichero acaba antes."""
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("fichero GGUF truncado")
    return data


def _read_scalar(handle, vtype: int):
    fmt, size = _SCALAR_FMT[vtype]
    return struct.unpack(fmt, _read_exact(handle, size))[0]


def _read_string(handle) -> str:
    length = struct.unpack("<Q", _read_exact(handle, 8))[0]
    return _read_exact(handle, length).decode("utf-8", errors="replace")


def _skip_array(handle, elem_type: int, count: int) -> None:
    """Salta el cuerpo de un array sin decodificarlo (lectura rápida de cabecera).

    Para tipos escalares avanza ``size*count`` bytes; para cadenas lee la longitud
    de cada una y la salta; para arrays anidados, recurre. Usa ``seek`` relativo.
    """
    if elem_type in _SCALAR_FMT:
        _, size = _SCALAR_FMT[elem_type]
        handle.seek(size * count, 1)
    elif elem_type == _STRING:
        for _ in range(count):
            longitud = struct.unpack("<Q", _read_exact(handle, 8))[0]
            handle.seek(longitud, 1)
    elif elem_type == _ARRAY:
        for _ in range(count):
            et = struct.unpack("<I", _read_exact(handle, 4))[0]
            n = struct.unpack("<Q", _read_exact(handle, 8))[0]
            _skip_array(handle, et, n)
    else:
        raise ValueError(f"tipo de array GGUF desconocido: {elem_type}")


def _read_value(handle, vtype: int, *, keep_full: bool = False, keep_limit=None,
                scalars_only: bool = False):
    """Lee un valor del tipo ``vtype``.

    Los arrays se resumen por defecto (longitud + primeros elementos como
    muestra), consumiendo el flujo completo para no descuadrar el parseo. Con
    ``keep_full`` se conservan los elementos (para ``/gguf tokens``); si
    ``keep_limit`` es un entero, sólo se guardan los primeros ``keep_limit``
    (aunque se siga leyendo el resto para avanzar el cursor), acotando la
    memoria al volcar arrays enormes como el vocabulario. Con ``scalars_only`` se
    saltan los arrays sin decodificarlos (lectura rápida de los metadatos
    escalares, p. ej. la arquitectura).
    """
    if vtype in _SCALAR_FMT:
        return _read_scalar(handle, vtype)
    if vtype == _STRING:
        return _read_string(handle)
    if vtype == _ARRAY:
        elem_type = struct.unpack("<I", _read_exact(handle, 4))[0]
        count = struct.unpack("<Q", _read_exact(handle, 8))[0]
        if scalars_only:
            _skip_array(handle, elem_type, count)
            return {"__array__": True, "len": count, "elem_type": elem_type,
                    "skipped": True}
        values = []
        sample = []
        for i in range(count):
            value = _read_value(handle, elem_type, keep_full=keep_full,
                                keep_limit=keep_limit)
            if keep_full:
                if keep_limit is None or len(values) < keep_limit:
                    values.append(value)
            elif i < 8:
                sample.append(value)
        result = {"__array__": True, "len": count, "elem_type": elem_type,
                  "sample": values[:8] if keep_full else sample}
        if keep_full:
            result["values"] = values
        return result
    raise ValueError(f"tipo de valor GGUF desconocido: {vtype}")


def read_scalar_metadata(path: str | Path) -> dict:
    """Lee sólo los metadatos escalares de un GGUF (rápido; salta los arrays).

    Útil para inspeccionar la arquitectura/nombre sin decodificar el vocabulario.
    Devuelve el mismo dict que :func:`read_gguf_metadata` pero con los arrays como
    marcadores ``{"__array__": True, "len": ..., "skipped": True}``.
    """
    return read_gguf_metadata(path, scalars_only=True)


def read_gguf_metadata(path: str | Path, *, full_arrays=None, array_limit=None,
                       scalars_only: bool = False) -> dict:
    """Lee la cabecera de ``path`` y devuelve versión, recuentos y metadatos.

    Args:
      path: Ruta al fichero GGUF.
      full_arrays: Opcional. Predicado ``clave -> bool``; para las claves donde
        devuelve True se conservan los elementos del array (no sólo una muestra).
        El resto de arrays se resumen para ahorrar memoria.
      array_limit: Opcional. Número máximo de elementos a conservar de cada array
        completo (el resto se lee pero se descarta). ``None`` = sin límite.

    Returns:
      ``{"version", "tensor_count", "kv_count", "metadata": {clave: valor}}``.
      Los valores de tipo array son un dict con ``len``/``sample`` (y ``values``
      si se pidió el contenido completo).

    Raises:
      ValueError: Si no es un GGUF válido o está truncado.
      OSError: Si el fichero no se puede abrir.
    """
    with open(path, "rb") as handle:
        if _read_exact(handle, 4) != GGUF_MAGIC:
            raise ValueError("no es un fichero GGUF (magic incorrecto)")
        version = struct.unpack("<I", _read_exact(handle, 4))[0]
        tensor_count = struct.unpack("<Q", _read_exact(handle, 8))[0]
        kv_count = struct.unpack("<Q", _read_exact(handle, 8))[0]
        metadata: dict = {}
        for _ in range(kv_count):
            key = _read_string(handle)
            vtype = struct.unpack("<I", _read_exact(handle, 4))[0]
            keep_full = bool(full_arrays and full_arrays(key))
            metadata[key] = _read_value(handle, vtype, keep_full=keep_full,
                                        keep_limit=array_limit,
                                        scalars_only=scalars_only)
    return {"version": version, "tensor_count": tensor_count,
            "kv_count": kv_count, "metadata": metadata}


def meaning_for_key(key: str, architecture: str | None) -> str:
    """Devuelve la explicación en español de ``key`` (o cadena vacía)."""
    if key in KEY_MEANINGS:
        return KEY_MEANINGS[key]
    if architecture and key.startswith(architecture + "."):
        generic = "{arch}." + key[len(architecture) + 1:]
        return KEY_MEANINGS.get(generic, "")
    return ""


def format_value(key: str, value, *, max_len: int = 80) -> str:
    """Convierte un valor de metadato en una cadena legible para mostrar."""
    if isinstance(value, dict) and value.get("__array__"):
        return f"[{value['len']} elementos]"
    if key == "general.file_type" and isinstance(value, int):
        return f"{value} ({FILE_TYPE_NAMES.get(value, 'desconocido')})"
    text = str(value)
    text = text.replace("\n", "\\n")
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def describe_gguf(path: str | Path, *, key_filter: str | None = None,
                  array_limit: int | None = None) -> dict:
    """Lee la cabecera y devuelve filas listas para mostrar con su significado.

    Args:
      path: Ruta al fichero GGUF.
      key_filter: Si se indica, sólo se devuelven las claves que lo contienen
        (sin distinguir mayúsculas). Además, las claves coincidentes que sean
        arrays se vuelcan COMPLETAS (campo ``array.values``), de modo que
        ``key_filter="tokens"`` lista todo el vocabulario.
      array_limit: Tope opcional de elementos a volcar por array (p. ej.
        ``/gguf tokens 100``). ``None`` = sin tope.

    Returns:
      ``{"version", "tensor_count", "kv_count", "architecture", "filter",
         "matched", "total", "rows": [{"key", "value", "meaning",
         "array": None | {"len", "elem_type", "values", "shown"}}]}``.
    """
    norm = key_filter.lower() if key_filter else None
    full_pred = (lambda key: norm in key.lower()) if norm else None
    info = read_gguf_metadata(path, full_arrays=full_pred, array_limit=array_limit)
    metadata = info["metadata"]
    architecture = metadata.get("general.architecture")

    rows = []
    for key, value in metadata.items():
        if norm is not None and norm not in key.lower():
            continue
        row = {
            "key": key,
            "value": format_value(key, value),
            "meaning": meaning_for_key(key, architecture),
            "array": None,
        }
        if isinstance(value, dict) and value.get("__array__") and "values" in value:
            row["array"] = {
                "len": value["len"],
                "elem_type": value["elem_type"],
                "values": value["values"],
                "shown": len(value["values"]),
            }
        rows.append(row)

    return {
        "version": info["version"],
        "tensor_count": info["tensor_count"],
        "kv_count": info["kv_count"],
        "architecture": architecture,
        "filter": key_filter,
        "matched": len(rows),
        "total": len(metadata),
        "rows": rows,
    }
