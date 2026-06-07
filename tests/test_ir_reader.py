"""Pruebas del lector de modelos OpenVINO IR (sin OpenVINO ni modelos reales)."""

import json
import tempfile
import unittest
from pathlib import Path

import ir_reader as ir


class DescribeIRTests(unittest.TestCase):
    def _ir_dir(self, *, config=None, gen=None, rt=True) -> Path:
        base = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(base, ignore_errors=True))
        (base / "openvino_model.bin").write_bytes(b"\x00" * 4096)
        if config is not None:
            (base / "config.json").write_text(json.dumps(config), encoding="utf-8")
        if gen is not None:
            (base / "generation_config.json").write_text(
                json.dumps(gen), encoding="utf-8")
        rt_xml = ('<rt_info><optimum><optimum_version value="1.20"/></optimum>'
                  '<nncf><weight_compression><bits value="4"/>'
                  '</weight_compression></nncf></rt_info>') if rt else ""
        (base / "openvino_model.xml").write_text(
            f'<net name="m"><layers/>{rt_xml}</net>', encoding="utf-8")
        return base

    def test_describe_basic(self):
        d = self._ir_dir(
            config={"model_type": "llama", "hidden_size": 4096,
                    "max_position_embeddings": 8192,
                    "architectures": ["LlamaForCausalLM"]},
            gen={"temperature": 0.7, "top_p": 0.9})
        info = ir.describe_ir(d)
        self.assertEqual(info["architecture"], "llama")
        self.assertGreaterEqual(len(info["files"]), 3)
        claves = {r["key"] for r in info["config_rows"]}
        self.assertIn("hidden_size", claves)
        # Significado en español presente para una clave conocida.
        meaning = next(r["meaning"] for r in info["config_rows"]
                       if r["key"] == "max_position_embeddings")
        self.assertIn("contexto", meaning.lower())
        self.assertEqual({r["key"] for r in info["gen_rows"]},
                         {"temperature", "top_p"})

    def test_rt_info_parsed(self):
        info = ir.describe_ir(self._ir_dir(config={"model_type": "llama"}))
        rt = {r["key"]: r["value"] for r in info["rt_info"]}
        self.assertEqual(rt.get("optimum/optimum_version"), "1.20")
        self.assertEqual(rt.get("nncf/weight_compression/bits"), "4")

    def test_architecture_from_architectures_list(self):
        info = ir.describe_ir(self._ir_dir(config={"architectures": ["Qwen2ForCausalLM"]}))
        self.assertEqual(info["architecture"], "Qwen2ForCausalLM")

    def test_rejects_non_ir(self):
        base = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(base, ignore_errors=True))
        with self.assertRaises(ValueError):
            ir.describe_ir(base)

    def test_doctype_rejected(self):
        # Un rt_info con DOCTYPE/ENTITY no se parsea (mitigación XXE).
        base = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(base, ignore_errors=True))
        (base / "openvino_model.bin").write_bytes(b"\x00")
        (base / "openvino_model.xml").write_text(
            '<net><rt_info><!DOCTYPE x><a value="1"/></rt_info></net>',
            encoding="utf-8")
        info = ir.describe_ir(base)
        self.assertEqual(info["rt_info"], [])


if __name__ == "__main__":
    unittest.main()
