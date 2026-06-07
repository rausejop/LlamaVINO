---
name: escritura-y-parches-windows
description: Escribir ficheros y aplicar parches (Git unified diff y Aider search/replace) de forma segura en Windows, sorteando rutas largas y ficheros bloqueados. Úsalo al guardar código generado por el LLM, aplicar diffs, o cuando una escritura falle por longitud de ruta o lock en Windows.
---

# Escritura de ficheros y parches seguros en Windows

Verificado en `edit_formats.py` (sin dependencia del modelo/OpenVINO). Aplicadores de
parches autocontenidos + escritura robusta en Windows.

## Gotchas de Windows (lo que de verdad falla)

- **Límite de 260 caracteres de ruta (`MAX_PATH`).** Prefija las rutas absolutas con
  `\\?\` (p. ej. `\\?\C:\muy\larga\ruta\fichero.py`) para usar la API de rutas largas.
  Aplícalo en escrituras, no solo en lecturas.
- **Ficheros bloqueados** (antivirus, indexador, editor abierto): la escritura/`replace`
  da `PermissionError`. Implementa **reintentos con backoff** corto antes de rendirte.
- Escribe con codificación **UTF-8 explícita**; no dependas de la codepage del sistema.
- Escritura atómica: escribe a un temporal y `os.replace` (atómico en el mismo volumen)
  para no dejar ficheros a medias si algo falla.

## Dos formatos de parche soportados

1. **Git Unified Diff** (`--- / +++ / @@`): aplica hunks por contexto. Tolera
   desplazamientos de línea razonables; valida que el contexto coincida.
2. **Aider search/replace**: bloques `<<<<<<< SEARCH … ======= … >>>>>>> REPLACE`.
   Busca el texto exacto y lo sustituye; ideal para ediciones que el LLM expresa como
   "esto por esto".

Elige según cómo emita el modelo el cambio; ambos confinados al workspace.

## Guardar código generado por el LLM

`extract_code_blocks` parsea bloques con triple backtick e **infiere el nombre** del
fichero (del info string, de un comentario inicial, o de la prosa anterior).
`write_workspace_file` lo escribe **confinado al workspace** (rechaza rutas que escapen).
En una UI, confirma antes de sobrescribir.

Relacionado: [[puente-ink-python-jsonld]] (mensajes `WriteFile`/`FileWritten`),
[[tui-chat-prompt-toolkit]] (autoguardado con confirmación).
