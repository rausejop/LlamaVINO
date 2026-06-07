---
name: puente-ink-python-jsonld
description: Conectar un frontend Node/Ink (React) con un backend Python de IA mediante un protocolo JSON-LD por líneas sobre stdio. Úsalo al diseñar la arquitectura híbrida, depurar el bridge, o añadir nuevos mensajes del protocolo.
---

# Puente Node/Ink ↔ Python por JSON-LD sobre stdio

Arquitectura **híbrida** verificada en LlamaVino: la UX rica vive en Node/Ink
(`ui/`), el motor del modelo vive en Python (OpenVINO GenAI). Hablan un **protocolo
JSON-LD por líneas** sobre stdio. El frontend hace `spawn("python", ["LlamaVino.py",
"--serve"])` y escribe/lee líneas JSON.

## Regla de oro del stream: stdout solo para protocolo

- **Toda** salida del protocolo va a **stdout**, una línea JSON por mensaje.
- **Todo** log incidental (telemetría, avisos, trazas) va a **stderr**. Si algo no
  protocolar se cuela en stdout, **corrompe el stream** y el frontend se rompe.
- En Python: `print(json.dumps(msg), flush=True)` a stdout; logs a `sys.stderr`.

## Forma de los mensajes (JSON-LD / Schema.org)

Cada mensaje lleva `@context` (de `context.jsonld`) y `@type`. Petición → respuesta(s):

- Carga: `Load` → `Loaded`; generación: `Generate` → varios `Token` + un cierre.
- Ficheros/código: `WriteFile`→`FileWritten`, `CodeOutline`, `ExtractSymbol`,
  `FileActionRequest`. Cabeceras: `GgufHeader`/`GgufInfo`, `IrHeader`/`IrInfo`.
- El backend mantiene `state["engine"]` (`openvino`/`llamacpp`) y despacha según el activo.

## Confinamiento de rutas (seguridad)

Toda operación de fichero pasa por `_confined_path`: resuelve contra el workspace y
**rechaza** rutas que escapen del directorio. No confíes en rutas del frontend.

## Cómo depurar sin TTY

`cd ui; node test-backend.mjs` ejercita el bridge Node↔Python sin terminal interactivo.
Para ver el protocolo crudo, lanza `python LlamaVino.py --serve` y escribe líneas JSON a
mano por stdin observando stdout (los logs saldrán por stderr, separados).

## Paridad de comandos

Ambas UIs (Ink y la TUI rich) reflejan el mismo set de comandos `/` adaptados al motor
local. Al añadir un comando, impleméntalo en las dos y en el handler `--serve`.

Relacionado: [[tui-chat-prompt-toolkit]] (la otra UI), [[ejecutar-gguf-en-iris-xe]].
