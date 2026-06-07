#!/usr/bin/env python
"""Motor de respaldo de LlamaVino basado en el binario de llama.cpp.

Cuando OpenVINO GenAI no puede con un modelo (arquitectura no soportada, o un
modelo más grande que la memoria que se quiere paginar desde disco con ``mmap``),
este motor delega en el ejecutable **precompilado** de
[llama.cpp](https://github.com/ggml-org/llama.cpp) (``llama-cli``), lanzándolo como
subproceso. No requiere compilador ni paquetes de Python: solo el binario.

Ventajas de esta ruta:

  * ``mmap`` está activo por defecto en llama.cpp → permite *cargar* modelos
    mayores que la RAM (se paginan desde disco; lento, pero arranca).
  * ``-ngl N`` descarga N capas a la GPU; con un binario **Vulkan** eso incluye la
    Intel Iris Xe.

Limitación honesta: se construye el prompt de forma genérica (no se aplica la
plantilla de chat exacta del modelo), así que la calidad conversacional es menor
que con la plantilla nativa. Es un motor de respaldo, no el principal.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

# Nombres del ejecutable de llama.cpp para generación one-shot, **en orden de
# preferencia**. Los builds recientes (b9xxx+) convirtieron ``llama-cli`` en una
# herramienta SIEMPRE interactiva que rechaza ``-no-cnv`` ("please use
# llama-completion instead") y se queda esperando entrada → cuelga el subproceso.
# Por eso se prefiere ``llama-completion`` (no interactivo) cuando existe; en
# builds antiguos no está y se recae en ``llama-cli``/``main`` (que sí completan).
_BINARY_NAMES = ("llama-completion", "llama-completion.exe",
                 "llama-cli", "llama-cli.exe", "main", "main.exe")

# Secuencias de escape ANSI (color) que llama-completion emite en stdout.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Nombres del servidor HTTP de llama.cpp (para el panel de candidatos).
_SERVER_NAMES = ("llama-server", "llama-server.exe", "server", "server.exe")
# Ubicaciones convencionales relativas a este repo.
_VENDOR_DIRS = ("vendor/llama.cpp", "llama.cpp", "bin")


class LlamaCppError(Exception):
    """No se encontró el binario o falló la ejecución de llama.cpp."""


def find_binary(explicit: str | None = None,
                names: tuple[str, ...] = _BINARY_NAMES,
                env_var: str = "LLAMA_CPP_BIN") -> str:
    """Localiza un ejecutable de llama.cpp (CLI o servidor).

    Orden de búsqueda: ``explicit`` -> variable de entorno ``env_var`` ->
    carpetas convencionales del repo -> ``PATH``.

    Args:
      explicit: Ruta concreta al binario (tiene prioridad).
      names: Nombres de ejecutable a buscar (CLI por defecto; servidor opcional).
      env_var: Variable de entorno que puede apuntar al binario.

    Returns:
      Ruta absoluta al ejecutable.

    Raises:
      LlamaCppError: Si no se encuentra ningún binario.
    """
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get(env_var)
    if env:
        candidates.append(env)
    here = Path(__file__).resolve().parent
    for vendor in _VENDOR_DIRS:
        for name in names:
            candidates.append(str(here / vendor / name))
    for name in names:
        found = shutil.which(name)
        if found:
            candidates.append(found)

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    cual = "servidor (llama-server)" if names is _SERVER_NAMES else "llama-cli"
    raise LlamaCppError(
        f"no se encontró el binario de {cual}. Define {env_var} o coloca el "
        "ejecutable en ./vendor/llama.cpp/."
    )


def find_server_binary(explicit: str | None = None) -> str:
    """Localiza el ejecutable ``llama-server`` (HTTP) de llama.cpp."""
    return find_binary(explicit, names=_SERVER_NAMES, env_var="LLAMA_SERVER_BIN")


def _crear_filtro_ansi():
    """Crea un filtro de códigos ANSI para un flujo leído por fragmentos.

    Devuelve ``filtrar(fragmento, final=False)`` que elimina las secuencias de
    color y conserva, entre llamadas, una secuencia de escape que quede partida en
    el borde de un fragmento (para no romperla ni dejar basura). Con
    ``final=True`` vacía cualquier resto pendiente.
    """
    carry = ""

    def filtrar(fragmento: str, final: bool = False) -> str:
        nonlocal carry
        datos = carry + fragmento
        carry = ""
        if not final:
            # Si al final hay un ESC sin terminar, lo guardamos para el siguiente.
            idx = datos.rfind("\x1b")
            if idx != -1 and not _ANSI.search(datos[idx:]):
                carry = datos[idx:]
                datos = datos[:idx]
        return _ANSI.sub("", datos)

    return filtrar


def detect_chat_format(model_path) -> str:
    """Deduce la plantilla de chat a partir de los metadatos del GGUF.

    Lee la clave estándar ``general.name`` (y ``general.architecture``) de la
    cabecera y devuelve ``"mistral"`` (``[INST] … [/INST]``), ``"gemma"``
    (``<start_of_turn>…``) o ``"generic"``. Si la cabecera no se puede leer, recae
    en el nombre del fichero.
    """
    texto = ""
    try:
        import gguf_reader  # import diferido: gguf_reader no depende de este módulo
        meta = gguf_reader.read_scalar_metadata(model_path)["metadata"]
        texto = " ".join(str(meta.get(clave, "")) for clave in
                         ("general.name", "general.architecture")).lower()
    except Exception:  # noqa: BLE001 - fichero ilegible: respaldo al nombre
        texto = Path(model_path).name.lower()
    if "mistral" in texto or "mixtral" in texto:
        return "mistral"
    if "gemma" in texto:
        return "gemma"
    if "phi" in texto:
        return "phi"
    if "qwen" in texto:
        return "qwen"
    return "generic"


def _build_prompt_qwen(history: list[dict]) -> str:
    """Construye el prompt con la plantilla ChatML de Qwen2.5/Qwen3.

    Formato: ``<|im_start|>rol\\n…<|im_end|>\\n`` por turno, cediendo al final el
    turno al asistente. Qwen tiene rol de sistema propio. llama.cpp añade el BOS.
    """
    salida = ""
    for turn in history:
        rol = turn["role"] if turn["role"] in ("system", "user", "assistant") else "user"
        salida += f"<|im_start|>{rol}\n{turn['content']}<|im_end|>\n"
    salida += "<|im_start|>assistant\n"
    return salida


def _build_prompt_phi(history: list[dict]) -> str:
    """Construye el prompt con la plantilla de Phi-3/Phi-3.5.

    Formato: ``<|system|>\\n…<|end|>\\n<|user|>\\n…<|end|>\\n<|assistant|>\\n…``.
    Phi sí tiene rol de sistema propio. llama.cpp añade el BOS.
    """
    etiquetas = {"system": "<|system|>", "user": "<|user|>",
                 "assistant": "<|assistant|>"}
    salida = ""
    for turn in history:
        etiqueta = etiquetas.get(turn["role"], "<|user|>")
        salida += f"{etiqueta}\n{turn['content']}<|end|>\n"
    salida += "<|assistant|>\n"
    return salida


def _build_prompt_gemma(history: list[dict]) -> str:
    """Construye el prompt con la plantilla de turnos de Gemma.

    Formato: ``<start_of_turn>user\\n{usuario}<end_of_turn>\\n<start_of_turn>model
    \\n{modelo}<end_of_turn>\\n…``. Gemma no tiene rol de sistema, así que se
    incrusta en el primer turno de usuario. llama.cpp añade el BOS.
    """
    sistema = " ".join(
        t["content"] for t in history if t["role"] == "system").strip()
    salida = ""
    primero = True
    for turn in history:
        rol = turn["role"]
        if rol == "system":
            continue
        if rol == "user":
            contenido = turn["content"]
            if primero and sistema:
                contenido = f"{sistema}\n\n{contenido}"
            primero = False
            salida += f"<start_of_turn>user\n{contenido}<end_of_turn>\n"
        elif rol == "assistant":
            salida += f"<start_of_turn>model\n{turn['content']}<end_of_turn>\n"
    salida += "<start_of_turn>model\n"
    return salida


def _build_prompt_mistral(history: list[dict]) -> str:
    """Construye el prompt con la plantilla de instrucciones de Mistral.

    Formato: ``[INST] {sistema}\\n\\n{usuario} [/INST] {asistente}</s>[INST] …``.
    El sistema se incrusta en el primer turno de usuario. llama.cpp añade el BOS.
    """
    sistema = " ".join(
        t["content"] for t in history if t["role"] == "system").strip()
    salida = ""
    primero = True
    for turn in history:
        rol = turn["role"]
        if rol == "system":
            continue
        if rol == "user":
            contenido = turn["content"]
            if primero and sistema:
                contenido = f"{sistema}\n\n{contenido}"
            primero = False
            salida += f"[INST] {contenido} [/INST]"
        elif rol == "assistant":
            salida += f" {turn['content']}</s>"
    return salida


def build_prompt(history: list[dict], fmt: str = "generic") -> str:
    """Construye un prompt de texto a partir del historial de chat.

    Con ``fmt="mistral"`` usa ``[INST]…[/INST]``; con ``fmt="gemma"`` los turnos
    ``<start_of_turn>…``; con ``fmt="phi"`` la plantilla ``<|user|>…<|end|>`` de
    Phi-3/3.5; en otro caso, un formato genérico con etiquetas de rol.
    """
    if fmt == "mistral":
        return _build_prompt_mistral(history)
    if fmt == "gemma":
        return _build_prompt_gemma(history)
    if fmt == "phi":
        return _build_prompt_phi(history)
    if fmt == "qwen":
        return _build_prompt_qwen(history)
    labels = {"system": "Sistema", "user": "Usuario", "assistant": "Asistente"}
    parts = [f"{labels.get(turn['role'], turn['role'])}: {turn['content']}" for turn in history]
    parts.append("Asistente:")
    return "\n".join(parts)


class LlamaCppEngine:
    """Genera texto con el binario de llama.cpp mediante subprocesos."""

    def __init__(
        self,
        model_path: str,
        *,
        binary: str | None = None,
        n_gpu_layers: int = 0,
        n_ctx: int = 4096,
        use_mmap: bool = True,
        chat_format: str = "auto",
    ) -> None:
        """Prepara el motor (valida que el binario y el modelo existan).

        Args:
          model_path: Ruta al fichero GGUF.
          binary: Ruta al ejecutable de llama.cpp (autodetecta si es None).
          n_gpu_layers: Capas a descargar a la GPU (``-ngl``). 0 = solo CPU.
          n_ctx: Tamaño de contexto.
          use_mmap: Si True (por defecto) el modelo se mapea con ``mmap`` y se
            pagina desde disco; si False se carga entero en memoria (``--no-mmap``).
          chat_format: Plantilla de chat (``"auto"`` la deduce del nombre;
            ``"mistral"`` usa ``[INST]…[/INST]``; ``"generic"`` por defecto).

        Raises:
          LlamaCppError: Si falta el binario o el modelo.
        """
        self.binary = find_binary(binary)
        self.model_path = str(model_path)
        if not Path(self.model_path).is_file():
            raise LlamaCppError(f"modelo no encontrado: {self.model_path}")
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.use_mmap = use_mmap
        self.chat_format = (chat_format if chat_format != "auto"
                            else detect_chat_format(self.model_path))
        self._proc = None  # subproceso activo (para poder cancelarlo)

    def _command(self, prompt: str, *, max_new_tokens, temperature, top_p, top_k) -> list[str]:
        """Compone la línea de comandos one-shot (llama-completion/llama-cli).

        ``--no-display-prompt`` deja en stdout sólo el texto generado y ``-no-cnv``
        fuerza el modo no conversacional; ambos los soporta ``llama-completion``
        (y el ``llama-cli`` antiguo), que es el binario que se prefiere.
        """
        command = [
            self.binary,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(max_new_tokens),
            "-c", str(self.n_ctx),
            "-ngl", str(self.n_gpu_layers),
            "--temp", str(temperature),
            "--top-p", str(top_p),
            "--top-k", str(top_k),
            "--no-display-prompt",  # stdout = solo el texto generado
            "-no-cnv",              # modo no conversacional (one-shot)
        ]
        if not self.use_mmap:
            # mmap está activo por defecto en llama.cpp; sólo lo desactivamos
            # explícitamente cuando se pide cargar el modelo entero en memoria.
            command.append("--no-mmap")
        return command

    def generate(
        self,
        history: list[dict],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        streamer=None,
    ) -> str:
        """Genera una respuesta para ``history`` y la devuelve como texto.

        Si se pasa ``streamer``, se le invoca con cada fragmento de texto a medida
        que llega (streaming). La salida de estadísticas de llama.cpp va a stderr y
        se descarta.

        Raises:
          LlamaCppError: Si el proceso de llama.cpp termina con error.
        """
        prompt = build_prompt(history, self.chat_format)
        command = self._command(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        proc = subprocess.Popen(  # noqa: S603 - array de args, sin shell
            command,
            stdin=subprocess.DEVNULL,  # ¡clave! evita que robe el stdin del terminal
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self._proc = proc
        chunks: list[str] = []
        filtrar = _crear_filtro_ansi()  # quita los códigos de color de llama-completion
        assert proc.stdout is not None
        try:
            for piece in iter(lambda: proc.stdout.read(64), ""):
                limpio = filtrar(piece)
                if limpio:
                    if streamer is not None:
                        streamer(limpio)
                    chunks.append(limpio)
            resto = filtrar("", final=True)
            if resto:
                if streamer is not None:
                    streamer(resto)
                chunks.append(resto)
            returncode = proc.wait()
        finally:
            self._proc = None
        if returncode not in (0, None) and not chunks:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise LlamaCppError(f"llama.cpp salió con código {returncode}: {stderr[-300:]}")
        texto = "".join(chunks).strip()
        # llama.cpp marca el fin de la secuencia con "[end of text]"; lo quitamos.
        if texto.endswith("[end of text]"):
            texto = texto[: -len("[end of text]")].rstrip()
        return texto

    def cancelar(self) -> None:
        """Detiene la generación en curso terminando el subproceso de llama-cli."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()


def _puerto_libre() -> int:
    """Devuelve un puerto TCP libre en localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _normaliza_candidatos(entry: dict) -> list[dict]:
    """Normaliza las probabilidades de un token a ``[{"word", "prob"}, ...]``.

    Tolera las distintas formas que ha usado llama.cpp: ``probs``/``top_logprobs``
    con ``tok_str``/``token``/``content`` y ``prob``/``logprob``.
    """
    probs = entry.get("probs") or entry.get("top_logprobs") or []
    salida: list[dict] = []
    for item in probs:
        palabra = item.get("tok_str") or item.get("token") or item.get("content") or ""
        prob = item.get("prob")
        if prob is None and "logprob" in item:
            try:
                prob = math.exp(float(item["logprob"]))
            except (TypeError, ValueError, OverflowError):
                prob = 0.0
        salida.append({"word": palabra, "prob": float(prob or 0.0)})
    return salida


class LlamaServerEngine:
    """Genera con ``llama-server`` (HTTP) exponiendo las probabilidades por token.

    A diferencia de :class:`LlamaCppEngine` (CLI), el servidor de llama.cpp puede
    devolver, por cada token, las palabras candidatas con su probabilidad
    (``n_probs``). Eso permite mostrar, antes de fijar cada palabra, la lista de
    candidatas y cuál se eligió. Usa ``mmap`` y reparto de capas (``-ngl``).
    """

    def __init__(
        self,
        model_path: str,
        *,
        binary: str | None = None,
        n_gpu_layers: int = 0,
        n_ctx: int = 4096,
        use_mmap: bool = True,
        n_probs: int = 5,
        chat_format: str = "auto",
        ready_timeout: float = 180.0,
    ) -> None:
        """Lanza ``llama-server`` y espera a que esté listo.

        Raises:
          LlamaCppError: Si falta el binario/modelo o el servidor no arranca.
        """
        self.binary = find_server_binary(binary)
        self.model_path = str(model_path)
        if not Path(self.model_path).is_file():
            raise LlamaCppError(f"modelo no encontrado: {self.model_path}")
        self.n_probs = max(1, int(n_probs))
        self.chat_format = (chat_format if chat_format != "auto"
                            else detect_chat_format(self.model_path))
        self._cancelar = False
        self.port = _puerto_libre()
        self.url = f"http://127.0.0.1:{self.port}"

        command = [
            self.binary, "-m", self.model_path,
            "-c", str(n_ctx), "-ngl", str(n_gpu_layers),
            "--host", "127.0.0.1", "--port", str(self.port),
        ]
        if not use_mmap:
            command.append("--no-mmap")
        self._proc = subprocess.Popen(  # noqa: S603 - array de args, sin shell
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            self._esperar_listo(ready_timeout)
        except LlamaCppError:
            self.close()
            raise

    def _esperar_listo(self, timeout: float) -> None:
        """Sondea ``/health`` hasta que el servidor responde ``ok`` o expira."""
        limite = time.monotonic() + timeout
        while time.monotonic() < limite:
            if self._proc.poll() is not None:
                raise LlamaCppError(
                    f"llama-server terminó al arrancar (código {self._proc.returncode})")
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=2) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read() or b"{}")
                        if data.get("status") == "ok":
                            return
            except Exception:  # noqa: BLE001 - aún arrancando/cargando
                pass
            time.sleep(0.3)
        raise LlamaCppError("llama-server no estuvo listo a tiempo")

    def generate(
        self,
        history: list[dict],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        streamer=None,
        on_candidates=None,
    ) -> str:
        """Genera por streaming; entrega texto y, por token, sus candidatas.

        ``streamer(texto)`` recibe cada token; ``on_candidates(cands, elegida)``
        recibe la lista ``[{"word","prob"}]`` y la palabra elegida de cada token.

        Raises:
          LlamaCppError: Si la petición HTTP al servidor falla.
        """
        payload = {
            "prompt": build_prompt(history, self.chat_format),
            "n_predict": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "n_probs": self.n_probs,
            "stream": True,
            "cache_prompt": True,
        }
        req = urllib.request.Request(
            f"{self.url}/completion",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        self._cancelar = False
        chunks: list[str] = []
        try:
            with urllib.request.urlopen(req) as resp:
                for cruda in resp:
                    if self._cancelar:
                        break
                    linea = cruda.decode("utf-8", "replace").strip()
                    if not linea.startswith("data:"):
                        continue
                    cuerpo = linea[5:].strip()
                    if cuerpo == "[DONE]":
                        break
                    try:
                        evento = json.loads(cuerpo)
                    except json.JSONDecodeError:
                        continue
                    cps = evento.get("completion_probabilities")
                    if cps:
                        for entrada in cps:
                            elegida = entrada.get("content", "")
                            if on_candidates is not None:
                                on_candidates(_normaliza_candidatos(entrada), elegida)
                            if streamer is not None and elegida:
                                streamer(elegida)
                            chunks.append(elegida)
                    else:
                        texto = evento.get("content", "")
                        if texto:
                            if streamer is not None:
                                streamer(texto)
                            chunks.append(texto)
                    if evento.get("stop"):
                        break
        except urllib.error.URLError as exc:
            raise LlamaCppError(f"fallo al hablar con llama-server: {exc}") from exc
        return "".join(chunks).strip()

    def cancelar(self) -> None:
        """Solicita detener la generación en curso (corta el streaming SSE)."""
        self._cancelar = True

    def close(self) -> None:
        """Detiene el proceso del servidor."""
        proc = getattr(self, "_proc", None)
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
