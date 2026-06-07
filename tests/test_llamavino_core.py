"""Pruebas de helpers de núcleo de LlamaVino (sin OpenVINO ni TTY)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import LlamaVino as lv  # noqa: E402


class ComandosTests(unittest.TestCase):
    def test_resolve_command_alias(self):
        self.assertEqual(lv._resolve_command("/modelo"), "/model")
        self.assertEqual(lv._resolve_command("/COLOR azul"), "/color")
        self.assertIsNone(lv._resolve_command("/desconocido"))

    def test_parsear_args_gguf(self):
        self.assertEqual(lv._parsear_args_gguf(""), (None, None))
        self.assertEqual(lv._parsear_args_gguf("rope"), ("rope", None))
        self.assertEqual(lv._parsear_args_gguf("tokens 100"), ("tokens", 100))
        self.assertEqual(lv._parsear_args_gguf("a b 5"), ("a b", 5))

    def test_tam_gb(self):
        self.assertEqual(lv._tam_gb("~8.5 GB"), 8.5)
        self.assertEqual(lv._tam_gb("0.80 GB"), 0.80)
        self.assertEqual(lv._tam_gb("sin número"), 0.0)


class CandidatasTests(unittest.TestCase):
    def test_comando_candidatas_resuelve(self):
        self.assertEqual(lv._resolve_command("/candidatas off"), "/candidatas")
        self.assertEqual(lv._resolve_command("/candidates"), "/candidatas")

    def test_print_historico_no_falla(self):
        import io

        from rich.console import Console
        hist = [{"pos": 1, "elegida": "Hola",
                 "lista": [{"word": "Hola", "prob": 0.9},
                           {"word": "Hello", "prob": 0.1}]}]
        buf = io.StringIO()
        lv._print_candidatas_hist(Console(file=buf), hist)
        self.assertIn("Hola", buf.getvalue())


class CodeBlockTests(unittest.TestCase):
    def test_filename_detection(self):
        texto = "Aquí tienes:\n```python hola.py\nprint('hola')\n```"
        bloques = lv.extract_code_blocks(texto)
        self.assertEqual(len(bloques), 1)
        self.assertEqual(bloques[0]["filename"], "hola.py")
        self.assertIn("print", bloques[0]["code"])

    def test_filename_from_comment(self):
        texto = "```python\n# script.py\nx = 1\n```"
        self.assertEqual(lv.extract_code_blocks(texto)[0]["filename"], "script.py")


class EngineHintTests(unittest.TestCase):
    def test_registry_hints(self):
        # bartowski Mistral GGUF -> respaldo llama.cpp; Llama -> openvino.
        self.assertEqual(lv._engine_hint("Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"),
                         "llamacpp")
        self.assertEqual(lv._engine_hint("Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
                         "openvino")


class EtiquetasTests(unittest.TestCase):
    def test_etiqueta_autor_y_motor(self):
        self.assertIn("magenta", lv.etiqueta_autor("unsloth"))
        self.assertIn("OpenVINO", lv.etiqueta_motor("openvino"))
        self.assertIn("respaldo", lv.etiqueta_motor("llamacpp"))


class IdentidadTests(unittest.TestCase):
    def test_temperatura_por_defecto_cero(self):
        self.assertEqual(lv.default_gen_settings()["temperature"], 0.0)
        self.assertEqual(lv.parse_args([]).temperature, 0.0)

    def test_sistema_identidad_llamavino(self):
        sis = lv.construir_sistema("models/gguf/phi-4-Q4_K_M.gguf")
        self.assertIn("LlamaVINO", sis)
        self.assertIn("NO digas que eres Claude", sis)
        self.assertIn("modelo fundacional", sis)

    def test_descripcion_modelo_ir(self):
        import json
        import tempfile
        base = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(base, ignore_errors=True))
        (base / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
        (base / "config.json").write_text(
            json.dumps({"model_type": "qwen2", "_name_or_path": "Qwen2.5-7B"}),
            encoding="utf-8")
        desc = lv.descripcion_modelo(base)
        self.assertIn("Qwen2.5-7B", desc)
        self.assertIn("qwen2", desc)


class BusquedaYReorgTests(unittest.TestCase):
    def test_params_de_nombre(self):
        self.assertEqual(lv._params_de_nombre("Qwen2.5-14B-Instruct"), 14.0)
        self.assertEqual(lv._params_de_nombre("TinyLlama-1.1B-Chat"), 1.1)
        self.assertIsNone(lv._params_de_nombre("phi-3.5-mini"))

    def test_filtra_candidatos_por_tamano(self):
        ids = ["a/M-32B", "b/M-14B", "c/M-7B", "d/M-1B"]
        repos = [r["repo"] for r in lv._filtra_candidatos(ids, min_b=1, max_b=14)]
        self.assertEqual(repos, ["b/M-14B", "c/M-7B", "d/M-1B"])

    def test_directorios_gguf_ir(self):
        self.assertEqual(lv.dir_gguf("models").name, "gguf")
        self.assertEqual(lv.dir_ir("models").name, "ir")


if __name__ == "__main__":
    unittest.main()
