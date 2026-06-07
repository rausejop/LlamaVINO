#!/usr/bin/env python
"""Aplicadores de parches para LlamaVino: Git Unified Diff y bloques Aider.

Este módulo es autocontenido y no depende del modelo ni de OpenVINO, de modo que
se puede probar de forma aislada. Implementa dos formatos estándar de edición:

  * **Git Unified Diff**: el formato de ``diff``/``git`` de toda la vida.
  * **Bloques Search & Replace estilo Aider**: ``<<<<<<< SEARCH`` / ``=======`` /
    ``>>>>>>> REPLACE``.

La escritura a disco respeta el entorno Windows: usa el prefijo de rutas largas
``\\\\?\\`` para superar el límite ``MAX_PATH`` (260) y reintenta con *backoff*
exponencial cuando otro proceso (compilador/IDE) tiene el fichero bloqueado.

Las funciones de aplicación en memoria (``apply_unified_diff``,
``apply_search_replace``) son puras: reciben el texto original y devuelven el
texto resultante, sin tocar el disco.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path


class PatchError(Exception):
    """Un parche no se pudo aplicar (contexto no encontrado o formato inválido)."""


# --------------------------------------------------------------------------- #
# Modelo de datos
# --------------------------------------------------------------------------- #


@dataclass
class Hunk:
    """Un bloque ``@@`` de un unified diff.

    Attributes:
      old_start: Línea inicial (1-based) en el fichero original.
      lines: Pares ``(op, texto)`` con ``op`` en ``{' ', '-', '+'}``.
    """

    old_start: int
    lines: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class FilePatch:
    """Parche unified diff de un único fichero."""

    old_path: str
    new_path: str
    hunks: list[Hunk] = field(default_factory=list)


@dataclass
class SearchReplaceBlock:
    """Un bloque Search & Replace estilo Aider."""

    path: str | None
    search: str
    replace: str


# --------------------------------------------------------------------------- #
# Unified diff (Git)
# --------------------------------------------------------------------------- #


def _parse_hunk_header(line: str) -> int:
    """Devuelve el número de línea inicial (1-based) de una cabecera ``@@``.

    Args:
      line: Cabecera del estilo ``@@ -12,7 +12,8 @@``.

    Returns:
      La línea inicial del lado original.

    Raises:
      PatchError: Si la cabecera no tiene el formato esperado.
    """
    # @@ -old_start,old_len +new_start,new_len @@
    try:
        old_part = line.split(" ")[1]  # "-12,7"
        old_start = int(old_part[1:].split(",")[0])
    except (IndexError, ValueError) as exc:
        raise PatchError(f"cabecera de hunk inválida: {line!r}") from exc
    return old_start


def parse_unified_diff(diff_text: str) -> list[FilePatch]:
    """Parsea un unified diff en una lista de parches por fichero.

    Acepta tanto diffs de ``git`` (con líneas ``diff --git``/``index``) como la
    salida de ``difflib.unified_diff``. Ignora el marcador
    ``\\ No newline at end of file``.

    Args:
      diff_text: El diff completo como texto.

    Returns:
      Una lista de :class:`FilePatch`.
    """
    patches: list[FilePatch] = []
    current: FilePatch | None = None
    hunk: Hunk | None = None

    for raw in diff_text.splitlines():
        if raw.startswith("--- "):
            current = FilePatch(old_path=_strip_path(raw[4:]), new_path="")
            hunk = None
        elif raw.startswith("+++ ") and current is not None:
            current.new_path = _strip_path(raw[4:])
            patches.append(current)
        elif raw.startswith("@@"):
            if current is None:
                raise PatchError("hunk sin cabecera de fichero (---/+++)")
            hunk = Hunk(old_start=_parse_hunk_header(raw))
            current.hunks.append(hunk)
        elif raw.startswith("\\"):
            continue  # "\ No newline at end of file"
        elif hunk is not None and raw[:1] in (" ", "+", "-"):
            hunk.lines.append((raw[0], raw[1:]))
        # cualquier otra línea (diff --git, index, etc.) se ignora

    return patches


def _strip_path(token: str) -> str:
    """Limpia una ruta de cabecera (``a/x.py\t2024`` -> ``x.py``)."""
    path = token.split("\t", 1)[0].strip()
    for prefix in ("a/", "b/"):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def _locate(original: list[str], block: list[str], hint: int) -> int | None:
    """Localiza ``block`` dentro de ``original`` cerca de ``hint`` (0-based).

    Permite un pequeño desajuste (fuzz) buscando hacia ambos lados, para tolerar
    diffs cuyos números de línea no encajen exactamente.

    Returns:
      El índice 0-based donde empieza el bloque, o None si no se encuentra.
    """
    if not block:
        return min(max(hint, 0), len(original))
    size = len(block)
    hint = max(0, min(hint, len(original)))
    if original[hint:hint + size] == block:
        return hint
    for delta in range(1, len(original) + 1):
        for cand in (hint - delta, hint + delta):
            if 0 <= cand <= len(original) - size and original[cand:cand + size] == block:
                return cand
    return None


def _apply_hunks(original: list[str], hunks: list[Hunk]) -> list[str]:
    """Aplica los hunks a una lista de líneas, devolviendo la lista resultante."""
    result: list[str] = []
    cursor = 0
    for hunk in hunks:
        old_block = [text for op, text in hunk.lines if op in (" ", "-")]
        pos = _locate(original, old_block, hunk.old_start - 1)
        if pos is None:
            raise PatchError(
                f"no se pudo localizar el contexto del hunk @@ {hunk.old_start}"
            )
        if pos < cursor:
            raise PatchError("hunks solapados o desordenados")
        result.extend(original[cursor:pos])

        idx = pos
        for op, text in hunk.lines:
            if op == " ":
                idx += 1
                result.append(text)
            elif op == "-":
                idx += 1
            else:  # "+"
                result.append(text)
        cursor = idx

    result.extend(original[cursor:])
    return result


def apply_unified_diff(diff_text: str, original: str) -> str:
    """Aplica un unified diff de un fichero a su texto original.

    Args:
      diff_text: El diff (un único fichero; si trae varios, se usa el primero).
      original: Texto original del fichero.

    Returns:
      El texto resultante tras aplicar el parche.

    Raises:
      PatchError: Si el diff está vacío o el contexto no encaja.
    """
    patches = parse_unified_diff(diff_text)
    if not patches:
        raise PatchError("el diff no contiene ningún parche de fichero")
    return _apply_file_patch_text(patches[0], original)


def _apply_file_patch_text(patch: FilePatch, original: str) -> str:
    """Aplica un :class:`FilePatch` preservando el salto de línea final."""
    trailing_nl = original.endswith("\n")
    lines = original.splitlines()
    patched = _apply_hunks(lines, patch.hunks)
    text = "\n".join(patched)
    if trailing_nl:
        text += "\n"
    return text


# --------------------------------------------------------------------------- #
# Bloques Search & Replace (Aider)
# --------------------------------------------------------------------------- #

_SEARCH = "<<<<<<< SEARCH"
_DIVIDER = "======="
_REPLACE = ">>>>>>> REPLACE"


def parse_search_replace(text: str) -> list[SearchReplaceBlock]:
    """Parsea uno o varios bloques Search & Replace estilo Aider.

    La ruta del fichero es la última línea no vacía (y que no sea una valla de
    código ```` ``` ````) antes del marcador ``<<<<<<< SEARCH``.

    Args:
      text: Texto que contiene los bloques.

    Returns:
      Lista de :class:`SearchReplaceBlock`.

    Raises:
      PatchError: Si un bloque queda sin cerrar.
    """
    lines = text.splitlines()
    blocks: list[SearchReplaceBlock] = []
    last_path: str | None = None
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith(_SEARCH):
            i += 1
            search: list[str] = []
            while i < len(lines) and lines[i].strip() != _DIVIDER:
                search.append(lines[i])
                i += 1
            if i >= len(lines):
                raise PatchError("bloque SEARCH sin '======='")
            i += 1  # salta el divisor
            replace: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith(_REPLACE):
                replace.append(lines[i])
                i += 1
            if i >= len(lines):
                raise PatchError("bloque sin '>>>>>>> REPLACE'")
            i += 1  # salta el cierre
            blocks.append(
                SearchReplaceBlock(
                    path=last_path,
                    search="\n".join(search),
                    replace="\n".join(replace),
                )
            )
        else:
            if stripped and not stripped.startswith("```"):
                last_path = stripped
            i += 1
    return blocks


def _flexible_replace(content: str, search: str, replace: str) -> str | None:
    """Reemplazo tolerante: compara líneas ignorando espacios al final.

    Returns:
      El contenido modificado, o None si no hay coincidencia.
    """
    content_lines = content.split("\n")
    search_lines = search.split("\n")
    norm = [line.rstrip() for line in search_lines]
    size = len(search_lines)
    for start in range(len(content_lines) - size + 1):
        window = [line.rstrip() for line in content_lines[start:start + size]]
        if window == norm:
            replaced = content_lines[:start] + replace.split("\n") + content_lines[start + size:]
            return "\n".join(replaced)
    return None


def apply_search_replace(content: str, block: SearchReplaceBlock) -> str:
    """Aplica un bloque Search & Replace a un contenido en memoria.

    Un ``search`` vacío significa "crear/sustituir todo el contenido" (fichero
    nuevo). Si el texto exacto no aparece, se intenta una coincidencia tolerante
    a espacios finales.

    Raises:
      PatchError: Si el bloque SEARCH no se encuentra en el contenido.
    """
    if block.search == "":
        return block.replace
    if block.search in content:
        return content.replace(block.search, block.replace, 1)
    flexible = _flexible_replace(content, block.search, block.replace)
    if flexible is not None:
        return flexible
    raise PatchError("el bloque SEARCH no se encontró en el contenido")


# --------------------------------------------------------------------------- #
# Escritura segura en Windows (rutas largas + backoff ante locking)
# --------------------------------------------------------------------------- #


def windows_long_path(path: Path) -> str:
    """Devuelve la ruta con prefijo ``\\\\?\\`` en Windows para superar MAX_PATH.

    En otros sistemas, o para rutas que no sean absolutas, devuelve la ruta tal
    cual. No transforma rutas UNC (``\\\\servidor\\...``).
    """
    text = str(path)
    if os.name == "nt" and path.is_absolute() and not text.startswith("\\\\"):
        return "\\\\?\\" + text
    return text


def read_text(path: str | Path) -> str:
    """Lee un fichero de texto UTF-8 con soporte de rutas largas."""
    target = Path(path)
    with open(windows_long_path(target), "r", encoding="utf-8", newline="") as handle:
        return handle.read()


def write_text_with_retry(
    path: str | Path,
    text: str,
    *,
    retries: int = 5,
    base_delay: float = 0.1,
) -> None:
    """Escribe texto UTF-8 reintentando ante bloqueos de fichero.

    Maneja el caso típico de Windows en el que el compilador o el IDE mantienen
    el fichero abierto: reintenta con *backoff* exponencial en vez de fallar.
    Preserva los saltos de línea (``newline=""``) y usa rutas largas.

    Args:
      path: Ruta de destino.
      text: Contenido a escribir.
      retries: Número de intentos antes de rendirse.
      base_delay: Retardo base en segundos (se duplica en cada intento).

    Raises:
      OSError: Si tras todos los reintentos el fichero sigue bloqueado.
    """
    target = Path(path)
    raw = windows_long_path(target)
    last_error: OSError | None = None
    for attempt in range(retries):
        try:
            with open(raw, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
            return
        except PermissionError as exc:  # fichero bloqueado por otro proceso
            last_error = exc
            time.sleep(base_delay * (2 ** attempt))
    assert last_error is not None
    raise last_error


def apply_patch_to_file(
    path: str | Path,
    patch_text: str,
    *,
    fmt: str = "unified",
) -> str:
    """Aplica un parche a un fichero en disco y guarda el resultado.

    Args:
      path: Fichero a modificar (debe existir salvo creación con Aider).
      patch_text: El parche (unified diff o bloques Aider).
      fmt: ``"unified"`` o ``"aider"``.

    Returns:
      El nuevo contenido escrito.

    Raises:
      PatchError: Formato desconocido o parche que no encaja.
    """
    target = Path(path)
    original = read_text(target) if target.exists() else ""
    if fmt == "unified":
        new_text = apply_unified_diff(patch_text, original)
    elif fmt == "aider":
        new_text = original
        for block in parse_search_replace(patch_text):
            new_text = apply_search_replace(new_text, block)
    else:
        raise PatchError(f"formato de parche desconocido: {fmt!r}")
    write_text_with_retry(target, new_text)
    return new_text
