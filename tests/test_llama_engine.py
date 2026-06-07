"""Tests de llama_engine (motor de respaldo llama.cpp por subproceso).

La generación real requiere el binario de llama.cpp; esos tests se omiten si no
está disponible. El resto valida la detección de binario y el armado del prompt
sin necesidad del ejecutable.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llama_engine as le  # noqa: E402


class BuildPromptTests(unittest.TestCase):
    def test_roles_and_turn_cession(self):
        history = [
            {"role": "system", "content": "Eres útil."},
            {"role": "user", "content": "Hola"},
            {"role": "assistant", "content": "¿Qué tal?"},
            {"role": "user", "content": "Bien"},
        ]
        prompt = le.build_prompt(history)
        self.assertIn("Sistema: Eres útil.", prompt)
        self.assertIn("Usuario: Hola", prompt)
        self.assertIn("Asistente: ¿Qué tal?", prompt)
        # Cede el turno al asistente al final.
        self.assertTrue(prompt.rstrip().endswith("Asistente:"))

    def test_unknown_role_passthrough(self):
        prompt = le.build_prompt([{"role": "tool", "content": "resultado"}])
        self.assertIn("tool: resultado", prompt)

    def test_mistral_format(self):
        history = [
            {"role": "system", "content": "Eres útil."},
            {"role": "user", "content": "Hola"},
            {"role": "assistant", "content": "¿Qué tal?"},
            {"role": "user", "content": "Bien"},
        ]
        prompt = le.build_prompt(history, "mistral")
        # El sistema se incrusta en el primer [INST]; turnos con [INST]…[/INST].
        self.assertIn("[INST] Eres útil.\n\nHola [/INST]", prompt)
        self.assertIn("¿Qué tal?</s>", prompt)
        self.assertTrue(prompt.rstrip().endswith("[INST] Bien [/INST]"))

    def test_gemma_format(self):
        history = [
            {"role": "system", "content": "Eres útil."},
            {"role": "user", "content": "Hola"},
            {"role": "assistant", "content": "Hey"},
            {"role": "user", "content": "Bien"},
        ]
        prompt = le.build_prompt(history, "gemma")
        self.assertIn("<start_of_turn>user\nEres útil.\n\nHola<end_of_turn>", prompt)
        self.assertIn("<start_of_turn>model\nHey<end_of_turn>", prompt)
        self.assertTrue(prompt.rstrip().endswith("<start_of_turn>model"))

    def test_phi_format(self):
        history = [
            {"role": "system", "content": "Eres útil."},
            {"role": "user", "content": "Hola"},
            {"role": "assistant", "content": "Hey"},
        ]
        prompt = le.build_prompt(history, "phi")
        self.assertIn("<|system|>\nEres útil.<|end|>", prompt)
        self.assertIn("<|user|>\nHola<|end|>", prompt)
        self.assertIn("<|assistant|>\nHey<|end|>", prompt)
        self.assertTrue(prompt.rstrip().endswith("<|assistant|>"))

    def test_qwen_format(self):
        history = [
            {"role": "system", "content": "Eres útil."},
            {"role": "user", "content": "Hola"},
        ]
        prompt = le.build_prompt(history, "qwen")
        self.assertIn("<|im_start|>system\nEres útil.<|im_end|>", prompt)
        self.assertIn("<|im_start|>user\nHola<|im_end|>", prompt)
        self.assertTrue(prompt.rstrip().endswith("<|im_start|>assistant"))

    def test_detect_chat_format_fallback_filename(self):
        # Sin cabecera legible (no existe), recae en el nombre del fichero.
        self.assertEqual(le.detect_chat_format("Mistral-7B-Instruct-v0.3.gguf"),
                         "mistral")
        self.assertEqual(le.detect_chat_format("google_gemma-3-4b-it.gguf"), "gemma")
        self.assertEqual(le.detect_chat_format("Phi-3.5-mini-instruct.gguf"), "phi")
        self.assertEqual(le.detect_chat_format("Qwen3-14B-Instruct.gguf"), "qwen")
        self.assertEqual(le.detect_chat_format("Llama-3.2-3B.gguf"), "generic")

    def test_detect_chat_format_from_metadata(self):
        import struct
        import tempfile

        def _s(t):
            b = t.encode()
            return struct.pack("<Q", len(b)) + b

        def _kv(k, v):
            return _s(k) + struct.pack("<I", 8) + _s(v)  # 8 = STRING

        # Nombre de fichero neutro, pero general.name dice "Mixtral".
        kvs = [_kv("general.architecture", "llama"),
               _kv("general.name", "Mixtral 8x7B Instruct")]
        data = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
        data += struct.pack("<Q", len(kvs)) + b"".join(kvs)
        tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        tmp.write(data)
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink())
        self.assertEqual(le.detect_chat_format(tmp.name), "mistral")


class FindBinaryTests(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("LLAMA_CPP_BIN", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["LLAMA_CPP_BIN"] = self._saved

    def test_explicit_path_used(self):
        # Un fichero real (este mismo test) sirve como "binario" para la prueba.
        here = str(Path(__file__).resolve())
        self.assertEqual(le.find_binary(here), here)

    def test_env_var_used(self):
        os.environ["LLAMA_CPP_BIN"] = str(Path(__file__).resolve())
        self.assertEqual(le.find_binary(), str(Path(__file__).resolve()))

    def test_missing_raises(self):
        os.environ["LLAMA_CPP_BIN"] = str(Path(__file__).parent / "no_existe_xyz.exe")
        with self.assertRaises(le.LlamaCppError):
            le.find_binary()


class EngineConstructionTests(unittest.TestCase):
    def test_missing_model_raises(self):
        # Usa este fichero como binario válido, pero un modelo inexistente.
        binary = str(Path(__file__).resolve())
        with self.assertRaises(le.LlamaCppError):
            le.LlamaCppEngine("modelo_inexistente.gguf", binary=binary)

    def test_command_includes_offload_and_sampling(self):
        binary = str(Path(__file__).resolve())
        model = str(Path(__file__).resolve())  # cualquier fichero existente
        engine = le.LlamaCppEngine(model, binary=binary, n_gpu_layers=20)
        cmd = engine._command("hola", max_new_tokens=64, temperature=0.5, top_p=0.8, top_k=30)
        self.assertIn("-ngl", cmd)
        self.assertIn("20", cmd)
        self.assertIn("--no-display-prompt", cmd)
        self.assertEqual(cmd[cmd.index("-n") + 1], "64")


class FiltroAnsiTests(unittest.TestCase):
    def test_quita_codigos_color(self):
        filtrar = le._crear_filtro_ansi()
        self.assertEqual(filtrar("\x1b[33mHola\x1b[0m mundo"), "Hola mundo")

    def test_secuencia_partida_entre_fragmentos(self):
        # Una secuencia ANSI partida en el borde no debe romperse ni dejar basura.
        filtrar = le._crear_filtro_ansi()
        a = filtrar("texto\x1b[")     # ESC sin terminar: se retiene
        b = filtrar("33mmás")          # completa la secuencia en el siguiente trozo
        self.assertEqual(a + b, "textomás")

    def test_esc_final_pendiente_se_vacia(self):
        filtrar = le._crear_filtro_ansi()
        parcial = filtrar("hola\x1b")
        resto = filtrar("", final=True)
        self.assertEqual(parcial, "hola")
        self.assertEqual(resto, "\x1b")  # ESC suelto al cerrar: se entrega tal cual

    def test_prefiere_llama_completion(self):
        # En la lista de nombres, llama-completion va antes que llama-cli.
        self.assertLess(le._BINARY_NAMES.index("llama-completion"),
                        le._BINARY_NAMES.index("llama-cli"))


class CancelarTests(unittest.TestCase):
    def test_llamacpp_cancelar_sin_proceso_es_seguro(self):
        # Sin generación en curso (_proc es None), cancelar() no debe fallar.
        binary = str(Path(__file__).resolve())
        model = str(Path(__file__).resolve())
        engine = le.LlamaCppEngine(model, binary=binary)
        self.assertIsNone(engine._proc)
        engine.cancelar()  # no lanza
        self.assertIsNone(engine._proc)

    def test_llamacpp_cancelar_termina_subproceso(self):
        # Simula una generación en curso con un subproceso real de larga duración.
        import subprocess
        binary = str(Path(__file__).resolve())
        model = str(Path(__file__).resolve())
        engine = le.LlamaCppEngine(model, binary=binary)
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(lambda: (proc.stdout.close(), proc.stderr.close()))
        engine._proc = proc
        engine.cancelar()
        proc.wait(timeout=10)
        self.assertIsNotNone(proc.poll())  # terminó

    def test_llamaserver_cancelar_activa_bandera(self):
        # Sin arrancar el servidor (evita el subproceso), solo la bandera.
        engine = le.LlamaServerEngine.__new__(le.LlamaServerEngine)
        engine._cancelar = False
        engine.cancelar()
        self.assertTrue(engine._cancelar)


if __name__ == "__main__":
    unittest.main(verbosity=2)
