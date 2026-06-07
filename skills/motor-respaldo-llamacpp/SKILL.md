---
name: motor-respaldo-llamacpp
description: Diagnosticar y operar el motor de respaldo llama.cpp por subproceso en Windows (arquitecturas que OpenVINO no lee, como Mistral). Úsalo cuando la generación con el binario de llama.cpp se cuelga, no devuelve texto, mete caracteres raros, o hay que elegir entre llama-cli y llama-completion.
---

# Motor de respaldo llama.cpp (subproceso) en Windows

Conocimiento verificado en este equipo (Windows 11, Iris Xe, 16 GB compartidos).
Aplica a `llama_engine.py`, que lanza el binario **precompilado** de llama.cpp como
subproceso cuando OpenVINO GenAI no puede con el modelo (p. ej. **Mistral v0.3**,
que da `invalid map<K,T> key` al leer el GGUF).

## El fallo que más cuesta diagnosticar: `llama-cli` ahora es interactivo

Los **builds recientes de llama.cpp (b9xxx+, verificado en b9542)** cambiaron
`llama-cli`: ya **no** es one-shot, es **siempre interactivo**.

- Rechaza `-no-cnv` con: `--no-conversation is not supported by llama-cli, please
  use llama-completion instead`.
- Carga el modelo y **se queda esperando entrada por stdin** → el subproceso
  **se cuelga** y parece que "el modelo no funciona". Era la causa real de que
  Mistral fallara (no la TUI).

**Solución:** usar **`llama-completion`** (no interactivo) para generación one-shot.
En `llama_engine._BINARY_NAMES` va **antes** que `llama-cli`; en builds antiguos sin
`llama-completion` se recae en `llama-cli`/`main`.

```powershell
# Comprobar qué soporta el binario antes de confiar en él:
.\vendor\llama.cpp\llama-cli.exe --help | Select-String "no-cnv|conversation"
# Si dice "please use llama-completion instead", el one-shot va por llama-completion.
```

## Cómo diagnosticar (sin TUI, aísla motor de interfaz)

Reproduce el problema **headless** en Python puro; si aquí cuelga/falla, el problema
es el motor, no prompt_toolkit:

```python
import llama_engine as le
print(le.find_binary())                 # debe resolver llama-completion(.exe)
eng = le.LlamaCppEngine("models/gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
                        n_gpu_layers=0, chat_format="auto")
print(eng.generate([{"role": "user", "content": "Di hola en una palabra"}],
                   max_new_tokens=16, temperature=0.0, streamer=lambda p: print(p, end="")))
```

Mistral genera "Hola!" en ~9 s, salida limpia, y **sale solo** (EXIT 0).

## Reglas firmes del subproceso

- **`stdin=subprocess.DEVNULL` SIEMPRE.** Sin esto el subproceso hereda el stdin del
  terminal y pelea con prompt_toolkit: caracteres basura al mover el ratón, cuelgues,
  Ctrl-C muerto.
- **Filtra los códigos ANSI de color.** `llama-completion` los emite en stdout aun con
  `--no-display-prompt`. `_crear_filtro_ansi()` los quita por fragmentos conservando
  una secuencia partida en el borde de una lectura.
- **Recorta el marcador final `[end of text]`** que llama.cpp añade al terminar.
- **Cancelación:** guarda el `Popen` en `self._proc` y expón `cancelar()` que llama a
  `proc.terminate()`; así Ctrl-C/Esc en la TUI recuperan el control sin matar el proceso
  principal.

## Plantillas de chat (one-shot no aplica la del modelo automáticamente)

`detect_chat_format(path)` lee `general.name`/`general.architecture` del GGUF y elige:
`mistral` (`[INST] … [/INST]`), `gemma` (`<start_of_turn>`), `phi` (`<|user|>…<|end|>`),
`qwen` (ChatML `<|im_start|>`) o `generic`. Pásalo como `chat_format="auto"`.

Relacionado: [[ejecutar-gguf-en-iris-xe]] para la ruta nativa OpenVINO.
