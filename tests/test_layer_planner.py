"""Pruebas del planificador de reparto de capas (sin GPU ni modelo reales)."""

import struct
import tempfile
import unittest
from pathlib import Path

import layer_planner as lp


def _string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_string(key: str, value: str) -> bytes:
    import gguf_reader as gr
    return _string(key) + struct.pack("<I", gr._STRING) + _string(value)


def _kv_uint32(key: str, value: int) -> bytes:
    import gguf_reader as gr
    return _string(key) + struct.pack("<I", gr._UINT32) + struct.pack("<I", value)


def _build_gguf(kv_blobs: list[bytes], *, padding: int = 0) -> bytes:
    header = b"GGUF" + struct.pack("<I", 3)
    header += struct.pack("<Q", 0) + struct.pack("<Q", len(kv_blobs))
    return header + b"".join(kv_blobs) + (b"\x00" * padding)


class PlanOffloadTests(unittest.TestCase):
    def _write(self, data: bytes) -> Path:
        tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        tmp.write(data)
        tmp.close()
        path = Path(tmp.name)
        self.addCleanup(path.unlink)
        return path

    def _model(self, block_count=32, padding=0):
        return self._write(_build_gguf([
            _kv_string("general.architecture", "llama"),
            _kv_uint32("llama.block_count", block_count),
            _kv_uint32("llama.embedding_length", 4096),
            _kv_uint32("llama.attention.head_count", 32),
            _kv_uint32("llama.attention.head_count_kv", 8),
        ], padding=padding))

    def test_no_gpu_means_cpu_only(self):
        # vram_total=0 fuerza el caso "sin VRAM utilizable" (None autodetectaría).
        plan = lp.plan_offload(self._model(), vram_total=0, free_ram=8 << 30)
        self.assertEqual(plan.n_gpu_layers, 0)
        self.assertFalse(plan.fits_full_gpu)
        self.assertIn("CPU", plan.reason)

    def test_partial_split_when_vram_is_tight(self):
        # KV por capa = 2 * 4096 * (4096*8/32) * 2 = 2*4096*1024*2 = 16 MiB aprox.
        plan = lp.plan_offload(
            self._model(block_count=32), n_ctx=4096,
            vram_total=200 * 1024 * 1024, free_ram=16 << 30,
            vram_usable_fraction=1.0, vram_overhead_bytes=0,
        )
        self.assertGreater(plan.n_gpu_layers, 0)
        self.assertLess(plan.n_gpu_layers, 32)
        self.assertFalse(plan.fits_full_gpu)
        self.assertIn("Reparto", plan.reason)

    def test_full_offload_when_vram_is_large(self):
        plan = lp.plan_offload(
            self._model(block_count=32), n_ctx=4096,
            vram_total=64 << 30, free_ram=16 << 30,
        )
        self.assertEqual(plan.n_gpu_layers, 32)
        self.assertTrue(plan.fits_full_gpu)

    def test_layer_size_estimate_scales_with_context(self):
        meta = {
            "general.architecture": "llama",
            "llama.block_count": 32,
            "llama.embedding_length": 4096,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
        }
        small = lp.estimate_layer_sizes(meta, file_size=320, n_ctx=1024)
        big = lp.estimate_layer_sizes(meta, file_size=320, n_ctx=8192)
        self.assertEqual(small["block_count"], 32)
        self.assertGreater(big["kv_layer_bytes"], small["kv_layer_bytes"])


if __name__ == "__main__":
    unittest.main()
