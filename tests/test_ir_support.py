"""Pruebas del soporte de directorios OpenVINO IR en LlamaVino (sin OpenVINO)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import LlamaVino as lv  # noqa: E402


class IRSupportTests(unittest.TestCase):
    def _ir_dir(self, ctx=None) -> Path:
        base = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(base, ignore_errors=True))
        ir = base / "mi-modelo-int4-ov"
        ir.mkdir()
        (ir / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
        (ir / "openvino_model.bin").write_bytes(b"\x00" * 2048)
        if ctx is not None:
            (ir / "config.json").write_text(
                json.dumps({"max_position_embeddings": ctx}), encoding="utf-8")
        return ir

    def test_es_modelo_ir(self):
        ir = self._ir_dir()
        self.assertTrue(lv.es_modelo_ir(ir))
        self.assertFalse(lv.es_modelo_ir(ir.parent))     # dir sin .xml
        self.assertFalse(lv.es_modelo_ir(ir / "openvino_model.xml"))  # un fichero

    def test_engine_hint_ir_es_openvino(self):
        self.assertEqual(lv._engine_hint(self._ir_dir()), "openvino")

    def test_scan_local_ir(self):
        ir = self._ir_dir()
        encontrados = lv.scan_local_ir(ir.parent)
        self.assertIn(ir, encontrados)

    def test_menu_incluye_ir(self):
        ir = self._ir_dir()
        filas = lv._build_model_menu(ir.parent)
        ir_rows = [r for r in filas if r["path"] == ir]
        self.assertEqual(len(ir_rows), 1)
        self.assertEqual(ir_rows[0]["engine"], "openvino")

    def test_contexto_modelo_desde_config(self):
        self.assertEqual(lv._contexto_modelo(self._ir_dir(ctx=8192)), 8192)
        self.assertIsNone(lv._contexto_modelo(self._ir_dir()))  # sin config.json

    def test_menu_publisher_y_destacado(self):
        base = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(base, ignore_errors=True))
        filas = {r["name"]: r for r in lv._build_model_menu(base)}
        # El registro deriva el autor del repo y marca el unsloth destacado.
        self.assertEqual(filas["qwen2.5-14b"]["publisher"], "bartowski")
        unsloth = filas["unsloth-qwen2.5-coder-14b"]
        self.assertEqual(unsloth["publisher"], "unsloth")
        self.assertTrue(unsloth["highlight"])

    def test_recomendacion_ordena_y_marca(self):
        base = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(base, ignore_errors=True))
        rows = lv._build_model_menu(base)
        # El primero es el recomendado #1 (qwen2.5-7b-ir).
        self.assertEqual(rows[0]["name"], "qwen2.5-7b-ir")
        self.assertEqual(rows[0]["recomendacion"], 1)
        # Las prioridades asignadas salen en orden ascendente y contiguo al inicio.
        ranks = [r["recomendacion"] for r in rows if r.get("recomendacion")]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual(ranks[:3], [1, 2, 3])

    def test_flag_validation(self):
        self.assertIsNone(lv._validar_args(lv.parse_args([])))
        self.assertIsNotNone(lv._validar_args(lv.parse_args(["--temperature", "9"])))
        self.assertIsNotNone(lv._validar_args(lv.parse_args(["--n-gpu-layers", "x"])))
        self.assertIsNone(lv._validar_args(lv.parse_args(["--n-gpu-layers", "20"])))

    def test_ov_plugin_config(self):
        cpu = lv.ov_plugin_config("CPU")
        gpu = lv.ov_plugin_config("GPU")
        self.assertIn("CACHE_DIR", cpu)          # caché de compilación siempre
        self.assertNotIn("KV_CACHE_PRECISION", cpu)
        self.assertIn("KV_CACHE_PRECISION", gpu)  # desactiva compresión KV en GPU

    def test_ajustes_round_trip(self):
        import os
        prev = Path.cwd()
        base = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(base, ignore_errors=True))
        os.chdir(base)
        self.addCleanup(lambda: os.chdir(prev))
        self.assertEqual(lv.cargar_ajustes(), {})  # sin fichero -> vacío
        lv.guardar_ajustes({"temperature": 1.1, "top_k": 33}, "#d75f5f")
        datos = lv.cargar_ajustes()
        self.assertEqual(datos["settings"]["temperature"], 1.1)
        self.assertEqual(datos["color"], "#d75f5f")


if __name__ == "__main__":
    unittest.main()
