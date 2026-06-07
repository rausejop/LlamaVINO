#!/usr/bin/env python
"""Cliente mínimo de Model Context Protocol (MCP) para LlamaVino.

Implementa la interfaz de cliente del estándar
[MCP](https://modelcontextprotocol.io/) sobre su transporte **stdio**: mensajes
JSON-RPC 2.0 delimitados por líneas. Cubre el ciclo básico que necesita un agente:

  1. ``initialize`` (handshake) + notificación ``notifications/initialized``.
  2. ``tools/list`` para descubrir las herramientas del servidor.
  3. ``tools/call`` para invocar una herramienta.

No depende de ningún SDK externo. El servidor se lanza como subproceso con
argumentos en array (sin shell) para evitar inyección de comandos.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

PROTOCOL_VERSION = "2024-11-05"


class MCPError(Exception):
    """Fallo en el diálogo con un servidor MCP."""


class MCPClient:
    """Cliente JSON-RPC 2.0 que conversa con un servidor MCP por stdio.

    Úsalo como gestor de contexto para garantizar el cierre del subproceso::

        with MCPClient([sys.executable, "servidor.py"]) as cli:
            cli.initialize()
            tools = cli.list_tools()
    """

    def __init__(self, command: list[str], *, cwd: str | None = None) -> None:
        """Lanza el servidor MCP.

        Args:
          command: Comando y argumentos del servidor (array, sin shell).
          cwd: Directorio de trabajo del servidor.
        """
        self.proc = subprocess.Popen(  # noqa: S603 - array de args, sin shell
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._next_id = 0
        self.server_info: dict[str, Any] | None = None

    # -- transporte -------------------------------------------------------- #

    def _send(
        self,
        method: str,
        params: dict | None = None,
        *,
        notification: bool = False,
    ) -> Any:
        """Envía un mensaje JSON-RPC; devuelve el ``result`` salvo notificación."""
        if self.proc.stdin is None:  # pragma: no cover - defensivo
            raise MCPError("el subproceso no tiene stdin")
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        if not notification:
            self._next_id += 1
            message["id"] = self._next_id
        self.proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        if notification:
            return None
        return self._read_result(self._next_id)

    def _read_result(self, expected_id: int) -> Any:
        """Lee líneas hasta encontrar la respuesta con el ``id`` esperado."""
        assert self.proc.stdout is not None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise MCPError("el servidor MCP cerró la conexión")
            line = line.strip()
            if not line:
                continue
            message = json.loads(line)
            if message.get("id") != expected_id:
                continue  # ignora notificaciones u otras respuestas
            if "error" in message:
                raise MCPError(message["error"].get("message", "error MCP"))
            return message.get("result")

    # -- API MCP ----------------------------------------------------------- #

    def initialize(self, client_name: str = "llamavino") -> dict:
        """Realiza el handshake ``initialize`` y guarda ``server_info``."""
        result = self._send(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": client_name, "version": "1.0"},
            },
        )
        self.server_info = result.get("serverInfo")
        self._send("notifications/initialized", notification=True)
        return result

    def list_tools(self) -> list[dict]:
        """Devuelve la lista de herramientas (`tools/list`)."""
        return self._send("tools/list", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        """Invoca una herramienta (`tools/call`) y devuelve su resultado."""
        return self._send("tools/call", {"name": name, "arguments": arguments or {}})

    def close(self) -> None:
        """Cierra los pipes y espera (o mata) el subproceso."""
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 - el cierre nunca debe propagar
            self.proc.kill()
        finally:
            for stream in (self.proc.stdout, self.proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:  # noqa: BLE001
                        pass

    def __enter__(self) -> "MCPClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
