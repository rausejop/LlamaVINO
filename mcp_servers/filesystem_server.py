#!/usr/bin/env python
"""Servidor MCP de ejemplo: acceso de solo lectura a un sistema de ficheros.

Implementa el lado servidor de [MCP](https://modelcontextprotocol.io/) sobre
stdio (JSON-RPC 2.0 por líneas) y expone dos herramientas:

  * ``list_dir(path)``  — lista las entradas de un directorio.
  * ``read_file(path)`` — lee un fichero de texto.

Todas las rutas se **confinan** a la raíz pasada con ``--root``: cualquier intento
de salir de ella (p. ej. con ``..``) se rechaza. Es un ejemplo deliberadamente
pequeño para demostrar la interfaz de cliente `mcp_client.MCPClient`.

Uso:
    python filesystem_server.py --root C:\\ruta\\permitida
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "list_dir",
        "description": "Lista las entradas (ficheros y carpetas) de un directorio.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Ruta a listar."}},
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": "Lee y devuelve el contenido de texto de un fichero.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Fichero a leer."}},
            "required": ["path"],
        },
    },
]


def _confine(root: Path, path_str: str) -> Path:
    """Resuelve ``path_str`` y verifica que no escapa de ``root``.

    Raises:
      ValueError: Si la ruta queda fuera de la raíz permitida.
    """
    target = Path(path_str)
    if not target.is_absolute():
        target = root / target
    target = target.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"ruta fuera de la raíz permitida: {path_str}")
    return target


def _text_result(text: str) -> dict:
    """Empaqueta texto en el formato de resultado de `tools/call`."""
    return {"content": [{"type": "text", "text": text}], "isError": False}


def handle(method: str, params: dict, root: Path) -> Any:
    """Despacha un método JSON-RPC de MCP y devuelve su ``result``.

    Raises:
      ValueError: Método o herramienta desconocidos, o ruta no permitida.
    """
    if method == "initialize":
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "filesystem-ejemplo", "version": "1.0"},
        }
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        name = params["name"]
        arguments = params.get("arguments", {})
        if name == "list_dir":
            target = _confine(root, arguments["path"])
            entries = sorted(
                p.name + ("/" if p.is_dir() else "") for p in target.iterdir()
            )
            return _text_result("\n".join(entries))
        if name == "read_file":
            target = _confine(root, arguments["path"])
            return _text_result(target.read_text(encoding="utf-8"))
        raise ValueError(f"herramienta desconocida: {name}")
    raise ValueError(f"método no soportado: {method}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Servidor MCP de ejemplo (filesystem).")
    parser.add_argument("--root", default=".", help="Raíz permitida (whitelist).")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    # Asegura UTF-8 en el transporte stdio (evita mojibake en Windows).
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        request_id = message.get("id")
        if request_id is None:
            continue  # notificación (p. ej. notifications/initialized): sin respuesta
        try:
            result = handle(message.get("method", ""), message.get("params") or {}, root)
            response = {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as exc:  # noqa: BLE001 - se reporta como error JSON-RPC
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32000, "message": str(exc)},
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
