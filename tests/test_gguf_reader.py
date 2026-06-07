"""Pruebas del lector de cabeceras GGUF (sin necesitar un modelo real)."""

import struct
import tempfile
import unittest
from pathlib import Path

import gguf_reader as gr


def _string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_string(key: str, value: str) -> bytes:
    return _string(key) + struct.pack("<I", gr._STRING) + _string(value)


def _kv_uint32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<I", gr._UINT32) + struct.pack("<I", value)


def _kv_string_array(key: str, values: list[str]) -> bytes:
    body = _string(key) + struct.pack("<I", gr._ARRAY)
    body += struct.pack("<I", gr._STRING) + struct.pack("<Q", len(values))
    for value in values:
        body += _string(value)
    return body


def _build_gguf(kv_blobs: list[bytes], *, version: int = 3) -> bytes:
    header = GGUF = b"GGUF"
    header += struct.pack("<I", version)
    header += struct.pack("<Q", 0)  # tensor_count
    header += struct.pack("<Q", len(kv_blobs))  # kv_count
    return header + b"".join(kv_blobs)


class ReadMetadataTests(unittest.TestCase):
    def _write(self, data: bytes) -> Path:
        tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        tmp.write(data)
        tmp.close()
        path = Path(tmp.name)
        self.addCleanup(path.unlink)
        return path

    def test_reads_scalars_strings_and_arrays(self):
        data = _build_gguf([
            _kv_string("general.architecture", "llama"),
            _kv_string("general.name", "Mi Modelo"),
            _kv_uint32("llama.block_count", 32),
            _kv_uint32("general.file_type", 15),
            _kv_string_array("tokenizer.ggml.tokens", ["a", "b", "c"]),
        ])
        path = self._write(data)

        info = gr.read_gguf_metadata(path)
        self.assertEqual(info["version"], 3)
        self.assertEqual(info["tensor_count"], 0)
        self.assertEqual(info["kv_count"], 5)
        meta = info["metadata"]
        self.assertEqual(meta["general.architecture"], "llama")
        self.assertEqual(meta["llama.block_count"], 32)
        self.assertTrue(meta["tokenizer.ggml.tokens"]["__array__"])
        self.assertEqual(meta["tokenizer.ggml.tokens"]["len"], 3)

    def test_rejects_non_gguf(self):
        path = self._write(b"NOPE" + b"\x00" * 16)
        with self.assertRaises(ValueError):
            gr.read_gguf_metadata(path)

    def test_describe_adds_meaning_and_formats_values(self):
        data = _build_gguf([
            _kv_string("general.architecture", "llama"),
            _kv_uint32("llama.context_length", 8192),
            _kv_uint32("general.file_type", 15),
            _kv_string_array("tokenizer.ggml.tokens", ["a", "b"]),
        ])
        path = self._write(data)

        desc = gr.describe_gguf(path)
        self.assertEqual(desc["architecture"], "llama")
        rows = {r["key"]: r for r in desc["rows"]}
        # La clave con prefijo de arquitectura obtiene su significado genérico.
        self.assertIn("contexto", rows["llama.context_length"]["meaning"].lower())
        # file_type se muestra con su nombre simbólico.
        self.assertEqual(rows["general.file_type"]["value"], "15 (Q4_K_M)")
        # Los arrays se resumen por longitud.
        self.assertEqual(rows["tokenizer.ggml.tokens"]["value"], "[2 elementos]")
        # Sin filtro, los arrays no se vuelcan (campo array vacío).
        self.assertIsNone(rows["tokenizer.ggml.tokens"]["array"])

    def test_filter_selects_matching_keys_only(self):
        data = _build_gguf([
            _kv_string("general.architecture", "llama"),
            _kv_uint32("llama.rope.dimension_count", 64),
            _kv_uint32("llama.rope.freq_base", 10000),
            _kv_uint32("llama.block_count", 32),
        ])
        path = self._write(data)

        desc = gr.describe_gguf(path, key_filter="rope")
        keys = {r["key"] for r in desc["rows"]}
        self.assertEqual(keys, {"llama.rope.dimension_count", "llama.rope.freq_base"})
        self.assertEqual(desc["matched"], 2)
        self.assertEqual(desc["total"], 4)
        self.assertEqual(desc["filter"], "rope")

    def test_filter_dumps_full_array(self):
        tokens = [f"tok{i}" for i in range(20)]
        data = _build_gguf([
            _kv_string("general.architecture", "llama"),
            _kv_string_array("tokenizer.ggml.tokens", tokens),
        ])
        path = self._write(data)

        desc = gr.describe_gguf(path, key_filter="tokens")
        self.assertEqual(len(desc["rows"]), 1)
        row = desc["rows"][0]
        self.assertIsNotNone(row["array"])
        self.assertEqual(row["array"]["len"], 20)
        # El array se vuelca COMPLETO (los 20 elementos, no sólo la muestra).
        self.assertEqual(row["array"]["values"], tokens)

    def test_array_limit_caps_dumped_values(self):
        tokens = [f"tok{i}" for i in range(50)]
        data = _build_gguf([
            _kv_string("general.architecture", "llama"),
            _kv_string_array("tokenizer.ggml.tokens", tokens),
        ])
        path = self._write(data)

        desc = gr.describe_gguf(path, key_filter="tokens", array_limit=10)
        row = desc["rows"][0]
        # Se conoce el total real pero sólo se vuelcan los primeros 10.
        self.assertEqual(row["array"]["len"], 50)
        self.assertEqual(row["array"]["shown"], 10)
        self.assertEqual(row["array"]["values"], tokens[:10])

    def test_no_filter_match_returns_empty_rows(self):
        data = _build_gguf([_kv_string("general.architecture", "llama")])
        path = self._write(data)
        desc = gr.describe_gguf(path, key_filter="inexistente")
        self.assertEqual(desc["rows"], [])
        self.assertEqual(desc["matched"], 0)

    def test_scalar_metadata_skips_arrays(self):
        data = _build_gguf([
            _kv_string("general.architecture", "llama"),
            _kv_string("general.name", "Mi Modelo"),
            _kv_uint32("llama.block_count", 32),
            _kv_string_array("tokenizer.ggml.tokens", ["a", "b", "c"]),
        ])
        path = self._write(data)

        meta = gr.read_scalar_metadata(path)["metadata"]
        self.assertEqual(meta["general.architecture"], "llama")
        self.assertEqual(meta["llama.block_count"], 32)
        # El array se salta (no se decodifica) pero se conserva la longitud.
        arr = meta["tokenizer.ggml.tokens"]
        self.assertTrue(arr["skipped"])
        self.assertEqual(arr["len"], 3)
        self.assertNotIn("sample", arr)


if __name__ == "__main__":
    unittest.main()
