"""Tests de MCP: cliente y servidor de ejemplo dialogando por stdio."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mcp_client  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SERVER = REPO / "mcp_servers" / "filesystem_server.py"


class _ServerFixture(unittest.TestCase):
    """Crea un workspace temporal y arranca el servidor MCP confinado a él."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "hola.txt").write_text("contenido ñ\n", encoding="utf-8")
        (self.root / "sub").mkdir()
        self.client = mcp_client.MCPClient(
            [sys.executable, str(SERVER), "--root", str(self.root)]
        )
        self.client.initialize()

    def tearDown(self):
        self.client.close()
        self.tmp.cleanup()


class HandshakeTests(_ServerFixture):
    def test_server_info(self):
        self.assertIsNotNone(self.client.server_info)
        self.assertEqual(self.client.server_info["name"], "filesystem-ejemplo")


class ToolDiscoveryTests(_ServerFixture):
    def test_list_tools(self):
        names = {t["name"] for t in self.client.list_tools()}
        self.assertEqual(names, {"list_dir", "read_file"})

    def test_tools_have_schema(self):
        tool = next(t for t in self.client.list_tools() if t["name"] == "read_file")
        self.assertIn("inputSchema", tool)
        self.assertEqual(tool["inputSchema"]["required"], ["path"])


class ToolCallTests(_ServerFixture):
    def test_list_dir(self):
        result = self.client.call_tool("list_dir", {"path": "."})
        text = result["content"][0]["text"]
        self.assertIn("hola.txt", text)
        self.assertIn("sub/", text)

    def test_read_file_utf8(self):
        result = self.client.call_tool("read_file", {"path": "hola.txt"})
        self.assertEqual(result["content"][0]["text"], "contenido ñ\n")

    def test_unknown_tool_errors(self):
        with self.assertRaises(mcp_client.MCPError):
            self.client.call_tool("borrar_todo", {})


class ConfinementTests(_ServerFixture):
    def test_escape_is_rejected(self):
        with self.assertRaises(mcp_client.MCPError):
            self.client.call_tool("read_file", {"path": "../../secreto.txt"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
