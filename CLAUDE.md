# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Author: © Rafael Ausejo Prieto, con ayuda de Claude Code.

## Goal (from specs.txt)

A Python script named **LlamaVino** that processes a quantized **GGUF** LLAMA model
file using the **OpenVINO** libraries, targeting an **Intel(R) Iris(R) Xe Graphics** GPU,
and applying the OpenVINO-backend optimizations from llama.cpp.

Key references:
- OpenVINO is installed via `python -m pip install openvino`.
- llama.cpp OpenVINO backend: https://github.com/ggml-org/llama.cpp

## Layout

- `LlamaVino.py` — Python CLI **and** the backend engine. Loads GGUF via
  `openvino_genai.LLMPipeline`; runs one-shot, an interactive `rich` TUI, or the
  `--serve` JSON-LD backend for the Ink UI.
- `ui/` — Node + Ink (React) frontend; the Claude Code-style TUI. Spawns
  `LlamaVino.py --serve` and talks JSON-LD over stdio. See `ui/README.md`.
- `ESTANDARES.md` — authoritative standards spec (JSON-LD/Schema.org, OTel GenAI, MCP,
  CommonMark safe subset, LSP, diffs, etc.) with Implementado/Roadmap status.
- `context.jsonld` — JSON-LD vocabulary (Schema.org-based) for the protocol.
- `system_prompt.md` — the agent security-harness core (Spanish, CommonMark).
- `skills/` — learned skills in **Agent Skills** format (`SKILL.md` + frontmatter).
- `edit_formats.py` — self-contained patch appliers (Git Unified Diff + Aider
  search/replace) with Windows-safe writes (`\\?\` long paths, lock backoff). Tests
  in `tests/test_edit_formats.py`. No dependency on the model/OpenVINO.
- `code_structure.py` — Tree-sitter code analysis: `list_symbols`, `extract_symbol`
  (exact function/class by qualified name), `outline`. Tests in
  `tests/test_code_structure.py`. tree-sitter import is lazy (optional dep).
- `ir_reader.py` — reads an OpenVINO **IR** model dir (`describe_ir`): files+sizes,
  `config.json`, `generation_config.json`, XML `rt_info`. No deps (defensive XML parse).
  Surfaced by `/ir`. Tests in `tests/test_ir_reader.py`.
- `perf_db.py` — SQLite (`models/llamavino.db`) of per-model performance: `registrar_carga`
  (load seconds), `registrar_generacion` (tok/s, running mean), `obtener`/`todos`/`borrar`.
  No deps. Tests in `tests/test_perf_db.py`.
- `mcp_client.py` + `mcp_servers/filesystem_server.py` — minimal MCP client and an
  example stdio server (JSON-RPC 2.0), no external SDK. Tests in `tests/test_mcp.py`.
  Server tools are root-confined. Wired into `--serve` (`McpListServers`/`McpListTools`/
  `McpCallTool`); pre-approved servers come from `mcp_config.json`. Not yet in the
  chat tool-loop (1B model).
- `llama_engine.py` — fallback engine driving the prebuilt **llama.cpp** binary
  (`llama-cli`) via subprocess: handles unsupported archs and oversized (mmap) or
  GPU-offload (`-ngl`, Vulkan) cases. No compiler/pip needed, just the binary
  (`LLAMA_CPP_BIN` or `vendor/llama.cpp/`). `use_mmap` (default on). Selected via
  `Load`'s `engine: "llamacpp"`/`"auto"`. Also `LlamaServerEngine` — spawns `llama-server`
  and streams `/completion` with `n_probs` to expose per-token candidate probabilities
  (`--engine llamaserver`). Tests in `tests/test_llama_engine.py`.
- `layer_planner.py` — real-time GPU/CPU layer-offload planner: reads the GGUF header,
  estimates per-layer weight + KV-cache cost, probes VRAM/free RAM, and returns an
  `OffloadPlan` (`n_gpu_layers`). No model load. Tests in `tests/test_layer_planner.py`.
- Backend protocol exposes both as JSON-LD requests (`FileActionRequest`,
  `CodeOutline`, `ExtractSymbol`), all directory-confined via `_confined_path`.
- `bench/benchmark.py` — backend benchmark (`llama-bench`): compares **llama.cpp · Vulkan
  (GPU)** vs **CPU** (`-ngl 0`) vs optional **SYCL** (`LLAMA_SYCL_BENCH`) vs optional
  **OpenVINO** (`--openvino`), over the local GGUFs. Saves `bench/resultados-*.md/.json`.
  Finding on this Iris Xe: **CPU wins token generation** (1B: ~25 vs ~15 t/s Vulkan vs 9.5
  OpenVINO; 7B: ~2× CPU) — the iGPU shares the memory bus, so generation is
  bandwidth-bound; GPU only wins prefill (`pp`). Drove the `--cpu` flag and the
  fastest-by-default behavior. `--registrar-perf` writes each model's CPU tg into
  `perf_db` (`motor=llamacpp`, GGUF). No external deps.
- `requirements.txt` — `openvino` + `openvino-genai` (>= 2025.1); OTel is optional.
- `specs.txt` — the original Spanish project brief.

Project-wide convention: **all of it is in Spanish** — UI text, code comments and
docstrings (`LlamaVino.py`, `gguf_reader.py` and the helper modules are fully Spanish);
JSON is **JSON-LD**; Markdown is **CommonMark**; Python follows PEP 8/257/484 + Google
Python Style Guide. Keep new code and comments in Spanish.

## Commands

```powershell
python -m pip install -r requirements.txt        # install Python deps
python LlamaVino.py                              # interactive rich TUI (default)
python LlamaVino.py --serve                       # JSON-LD backend (used by the Ink UI)
python LlamaVino.py --list-devices               # enumerate OpenVINO devices
python LlamaVino.py --download llama-3.2-3b      # fetch a model from Hugging Face
python LlamaVino.py --convert-ir Qwen/Qwen2.5-14B-Instruct  # HF -> OpenVINO IR int4
python LlamaVino.py -m model.gguf -p "prompt"    # one-shot generation (AUTO device)
python LlamaVino.py -m ./qwen2.5-14b-int4-ov -p "prompt"    # one-shot from an IR dir
python -m py_compile LlamaVino.py                 # syntax check
python -m unittest discover tests                 # run the test suite (edit_formats)

cd ui; npm install                                # install UI deps (once)
cd ui; npm start                                  # launch the Ink frontend
cd ui; node test-backend.mjs                      # test the Node<->Python bridge (no TTY)
```

## Architecture

- **Hybrid**: the rich UX (Ink/React, `ui/`) is a Node frontend; the model engine is
  Python (OpenVINO GenAI). They speak a **JSON-LD line protocol** over stdio
  (`serve_stdio` in `LlamaVino.py` ↔ `ui/source/backend.js`). Protocol output goes to
  stdout; all incidental logs go to stderr so they never corrupt the stream.
- **Run modes** in `LlamaVino.py`: one-shot (`-m`/`-p`), interactive `rich` TUI
  (`interactive_session`, default on a TTY), and `--serve` (the Ink backend).
- **Telemetry**: `_init_telemetry` sets up OpenTelemetry GenAI spans (console exporter
  on stderr) around generation; a no-op if the OTel SDK is not installed.
- **Chat context**: the rich TUI keeps a plain list of turns and rebuilds an
  `openvino_genai.ChatHistory` (prefixed with the system prompt) each `generate()` call. Do
  **not** use `start_chat()`/`finish_chat()` — deprecated in OpenVINO GenAI 2026.x.
- **Identity system prompt**: `construir_sistema(model_path)` builds the system message —
  "you are **LlamaVINO** (GGUF/llama.cpp + OpenVINO), not Claude/the base model; only if
  asked the *foundational model* answer with the real one" (`descripcion_modelo` reads
  GGUF `general.name`/arch or IR `config.json`) + `FILE_PRIMER`. Stored on the engine
  (`engine["sistema"]`, set in `crear_motor`) and prepended as a `system` turn for **all**
  engines in `_motor_generar`. Default **temperature is 0.0** (`default_gen_settings`,
  `--temperature`).
- **Chat slash commands**: both UIs mirror Claude Code's set, adapted to the local engine
  (`CHAT_COMMANDS`): `/help /clear /compact /config /model /save /cost /status /gguf /ir
  /gpu /cpu /color /doctor /mcp /candidatas /exit`. Both UIs show a filtered, arrow-navigable autocomplete menu
  when the input starts with `/` — the Ink UI via `slashMatches`/`SlashMenu`, the rich TUI
  via **prompt_toolkit** (`_crear_editor`: a `Completer` + `complete_while_typing` + a
  `Lexer`). Commands render in blue; `/color` offers its color options on Tab.
- **Rich TUI** (`ChatTUI`): a **full-screen** prompt_toolkit `Application`. `HSplit` layout —
  a scrolling **output** window (read-only `Buffer`, cursor pinned to end so it follows the
  conversation; mouse-wheel scrolls up to see history, `mouse_support=True`), an optional
  **candidates** window, then FIXED at the bottom: **top bar / prompt / bottom bar / status**.
  Bars are `Window(char="─", style=callable)` → full width, re-flow on resize; color driven
  by `tema` (`/color`). The prompt is `multiline=False` (visually wraps). Status `VSplit`:
  left `_status_izq` = **`LlamaVINO`** (fijo) `| ruta | modelo | Contexto: n% usado` y, con
  llama-server y la vista en vivo activa, el token candidato elegido (`cand: '…'`) entre el
  contexto y los tokens; right `_status_der` = **`i / j tokens`** (i = tokens en uso en vivo,
  j = máximo de la sesión = contexto). Ambos se actualizan **en tiempo real**. Generation runs in a background executor (`create_background_task` +
  `run_in_executor`); the streamer appends to the output and schedules redraws via
  `loop.call_soon_threadsafe` + `app.invalidate()`. Long informational commands
  (`/ir /gguf /help /status /cost /doctor /mcp /gpu`) use `_pager` (`run_in_terminal` +
  ENTER pause) so output is readable/scrollable on the normal terminal; `/config`/`/save`
  also use `run_in_terminal`; `/color` logs inline; `/model` exits with `accion="switch"`
  → `_SwitchModel`.
- **GPU KV-cache**: `ov_plugin_config(device)` sets `KV_CACHE_PRECISION=f16` on GPU when
  building the pipeline (`build_pipeline` and `--serve`). The GPU default (u8 KV-cache
  compression) breaks the chat `get_state` path ("get_state API is supported only when
  KV-cache compression is disabled").
- **Per-token candidates** (`--engine llamaserver`): `LlamaServerEngine` spawns `llama-server`
  and streams `/completion` with `n_probs`, so each token arrives with its top-k candidate
  words + probabilities. The Live view then shows a `_tabla_candidatos` (chosen word in green
  `→`, rest dim) above the streaming text. Needs the `llama-server` binary (`LLAMA_SERVER_BIN`
  or `vendor/llama.cpp/`); only on that engine (OpenVINO / `llama-cli` don't expose probs).
  `crear_motor` builds it; `cerrar_motor` stops the server on exit/switch. `--n-probs` (5).
  El panel **persiste tras generar** (muestra el último token, cabecera "(último token)")
  y se limpia al empezar la siguiente. `/candidatas [on|off]` alterna la **vista en vivo**
  (panel + token en la barra de estado; bandera `candidatas_vivo`, persistida en
  `.llamavino.json` y por defecto **on**); `/candidatas historico` revisa en el pager el
  **histórico por token** (`hist_candidatas`, registrado siempre en `_on_cands` aunque la
  vista esté apagada) vía `_print_candidatas_hist`.
- **Engine + layer offload**: `crear_motor` picks the engine and `_motor_generar` dispatches
  generation for both. `--engine {auto,openvino,llamacpp}` (default `auto`): `auto` routes a
  model that does **not** fit fully in VRAM to **llama.cpp** (with mmap) when a binary is
  present, else OpenVINO. `layer_planner.py` computes, in real time, how many layers go to
  the GPU (`-ngl`) vs CPU RAM from the GGUF size/`block_count`, the per-layer KV-cache cost,
  and the detected VRAM (`GPU_DEVICE_TOTAL_MEM_SIZE`) / free RAM. `--n-gpu-layers auto|N`,
  `--n-ctx`. `/gpu` shows the plan; `/status` shows engine + GPU/CPU layer split. llama.cpp
  uses `mmap` by default (`use_mmap`, `--no-mmap` only when disabled). Tests in
  `tests/test_layer_planner.py`. Under `--engine auto`, if OpenVINO **fails to read a GGUF**
  (e.g. `invalid map<K,T> key` for archs/metadata it can't ingest, like Mistral v0.3),
  `crear_motor` auto-falls-back to llama.cpp when a binary is present; otherwise it raises a
  clear message telling the user to install the binary or pick a llama-arch model. Each
  `MODELS` entry carries an `engine` hint (`openvino`/`llamacpp`); `_engine_hint(path)` maps a
  GGUF filename to it. Models live under `--models-dir` (default **`models/`**). The picker
  shows columns **Modelo · Autor · Tamaño · Formato (GGUF/IR) · Motor**, sorted by size
  (desc, `_tam_gb`). **Autor** comes from `repo.split("/")[0]` (`etiqueta_autor`, colored
  per publisher: bartowski/OpenVINO/unsloth/TheBloke/MaziyarPanahi/local). A model can set
  `highlight: True` to render specially (✦ magenta) — used for `unsloth-qwen2.5-coder-14b`,
  the most powerful unsloth model that runs natively on OpenVINO here. **Motor** via
  `etiqueta_motor`: green *OpenVINO* vs yellow *llama.cpp (respaldo)* vs dim *auto*. Under
  `auto`, a `llamacpp`-hinted model (e.g. Mistral v0.3) goes straight to llama.cpp, skipping
  the doomed OpenVINO attempt — both in `crear_motor` (rich TUI) and the `Load` handler (Ink).
- **Fastest-option auto-default + `--cpu`/`--gpu`/`/cpu`**: `resolver_velocidad(args, device)`
  picks the **fastest** config for this machine and **enables it by default**; `crear_motor`
  stores the decision in `engine["velocidad"]` (`{recomendado, forzar_cpu, es_rapida, auto,
  etiqueta_rapida, nota}`). `_gpu_es_integrada(device)` (OpenVINO `DEVICE_TYPE`/name) → on an
  **integrated GPU** (Iris Xe) the recommendation is **CPU** (generation is bandwidth-bound;
  see `bench/benchmark.py`). With nothing forced, it auto-selects CPU (`forzar_cpu` → `_ngl()`
  returns 0, OpenVINO `device="CPU"`, `auto` biases to **llama.cpp** when a binary exists;
  label `llama.cpp (solo CPU)`). Forcing a slower option (`--gpu`, `--device GPU`, explicit
  `--engine`/`--n-gpu-layers>0`) is honored but **warns** which option is fastest. `--cpu`
  forces CPU; `--gpu` forces GPU (mutually exclusive, checked in `_validar_args`). The `nota`
  is shown: one-shot → stderr; rich TUI → first transcript line (on start, `/model` switch and
  `/cpu` reload); `/status` adds *Opción más rápida* rows. Chat command **`/cpu on|off`**
  (`_cmd_cpu`) toggles the mode by reloading the engine via `accion="cpu"` (handled in
  `interactive_session`: sets `args.cpu`/`args.gpu`, re-creates the engine, re-enters keeping
  the conversation). Not wired into `--serve`/Ink (which don't call `crear_motor`).
  Local (non-registry) GGUFs are detected by reading the header cheaply
  (`gguf_reader.read_scalar_metadata`, which skips array bodies via `_skip_array`):
  `_detectar_motor_gguf` maps `general.architecture` (+ Mistral/Mixtral name check) to a
  motor (`OPENVINO_ARCHS = {llama, qwen2, phi3, gemma}`), cached by `(path, mtime)`. The
  picker shows it.
- **Binario llama.cpp one-shot (`llama-completion` > `llama-cli`)**: los builds recientes
  (b9xxx+) convirtieron `llama-cli` en una herramienta **siempre interactiva** que rechaza
  `-no-cnv` ("please use llama-completion instead") y se queda esperando entrada → el
  subproceso se cuelga (era el verdadero motivo de que **Mistral fallara**). Por eso
  `_BINARY_NAMES` prefiere **`llama-completion`** (no interactivo) y recae en `llama-cli`/
  `main` sólo en builds antiguos. `llama-completion` emite **códigos ANSI de color** en
  stdout: `_crear_filtro_ansi()` los quita por fragmentos (conservando una secuencia partida
  en el borde), y se recorta el marcador final `[end of text]`. Tests `FiltroAnsiTests`.
- **Chat templates for the fallback engine**: `llama_engine.build_prompt(history, fmt)`
  supports `fmt="mistral"` (`[INST] … [/INST]`), `fmt="gemma"` (`<start_of_turn>…`),
  `fmt="phi"` (`<|user|>…<|end|><|assistant|>`), `fmt="qwen"` (ChatML `<|im_start|>…`) and
  `"generic"`. `detect_chat_format(path)` reads the GGUF **`general.name`** (+
  `general.architecture`) from the header (falls back to the filename); both engines take
  `chat_format="auto"` and apply it.
- **Robustness / persistence / startup**: the TUI wraps `_procesar` (and broadens the
  generate/compact `except`) so engine errors show in-chat instead of crashing the event
  loop. Interactive settings + bar color persist to `.llamavino.json`
  (`cargar_ajustes`/`guardar_ajustes`, applied in `ChatTUI.__init__`, saved on
  `/config`/`/color`). `ov_plugin_config` sets `CACHE_DIR` to **`TEMP_DIR/ovcache`** to cache
  compiled models for fast reloads. `TEMP_DIR` = regenerable-temp base, default
  **`C:\temp\llamavino`** (override with env `LLAMAVINO_TEMP`) — kept **outside the repo** so
  the compiled-model cache (can reach 10–15 GB) doesn't fill the project disk and can be
  wiped freely (`rm -rf C:\temp\llamavino`). `_ajustar_hf()` lowers HF Hub log noise (and
  `HF_TOKEN` is used automatically). Top-level `README.md` + `.gitignore` added.
- **Arrow-key selectors**: `_picker_lista(lineas, get_help=…)` is a reusable full-screen
  prompt_toolkit list (Buffer + `cursorline`, scrolls; ↑/↓ + PageUp/Down, Enter, q/Esc) with
  a per-item help line. Used by `interactive_select_model` (with `_fila_modelo` columns) and
  `configure_settings` (each parameter shows its `GEN_PARAM_SPECS` help on highlight). Both
  fall back to a numbered prompt if no TTY. To avoid nesting apps, `/model` and `/config`
  **exit** `ChatTUI` (`accion="switch"/"config"`); `interactive_session` runs the picker and
  **re-enters** via `ChatTUI(..., estado=tui.exportar_estado())`, preserving the conversation,
  settings, color and stats (no process restart; `/config` keeps the same engine).
- **Last model + version + flag validation + downloads**: `interactive_select_model` marks
  and quick-selects the `last_model` (persisted via `guardar_ajustes(last_model=…)`, which now
  merges). `--version`/`__version__`. `_validar_args` checks flag ranges (temperature, top-p/k,
  n-ctx, penalties, `--n-gpu-layers auto|int`) before running. `download_model`/`_descargar_ir`
  print clear `⬇/✓` headers with repo, destination, final size and elapsed.
- **Recommendation ranking**: `RECOMENDACION` (alias → priority) ranks models for this
  machine (1 = best power/speed `qwen2.5-7b-ir`/`qwen2.5-7b`/`llama-3.1-8b`; then the 14B;
  then the fastest). The picker shows a **Rec.** column (`#1…`) and **sorts by priority**
  (ranked first, then the rest by size). Surfaced in rich + Ink + `_models_payload`.
- **Storage layout + perf + CRUD**: models live in `<models-dir>/gguf` and `<models-dir>/ir`
  (`dir_gguf`/`dir_ir`; scans still read the base dir for back-compat; `_ruta_registro`
  resolves registry entries in either). The picker shows a **Rend.** column (tok/s · load s)
  from `perf_db`, recorded by `build_pipeline`/`crear_motor` (load) and `ChatTUI._generar`
  (tok/s). Picker actions: **b** = buscar new models on HF (`buscar_modelos` by format/size,
  then pick + download `_ficheros_gguf`/`_descargar_ir`), **d** = `borrar_modelo_disco`
  (delete file/dir + perf row). Load = Enter; unload = `/model` switch or `/exit`.
- **Model registry** (`MODELS`, bartowski repos, ordered most→least powerful, ≤~14B to fit
  16 GB): native OpenVINO (llama/qwen2/phi3) — `qwen2.5-14b`, `phi-4`, `qwen2.5-coder-14b`,
  `phi-3-medium`, `llama-3.1-8b`, `qwen2.5-7b`, `qwen2.5-coder-7b`, `phi-3.5-mini`,
  `llama-3.2-3b`, `qwen2.5-3b`, `llama-3.2-1b`; llama.cpp fallback — `mistral-7b`,
  `gemma-3-4b`, `gemma-3-1b`. >14B (32B/70B) omitted (don't fit). Each carries an `engine`
  hint shown in the picker.
- **GGUF header**: `gguf_reader.py` parses the GGUF metadata block (no deps);
  `describe_gguf(path, key_filter=, array_limit=)` returns rows with each key's value + a
  Spanish meaning. `/gguf` shows the summary; `/gguf <filtro>` filters keys; `/gguf tokens
  [N]` dumps a matching array (optionally capped at N). Rich table in the TUI;
  `GgufHeader`/`GgufInfo` JSON-LD for the Ink UI. Tests in `tests/test_gguf_reader.py`.
- **File writing**: `extract_code_blocks` parses fenced blocks and infers a filename (info
  string, leading comment, or preceding prose); `write_workspace_file` writes it confined to
  the workspace. The rich TUI auto-saves named blocks (with confirmation) and exposes
  `/save [path]`; the Ink UI auto-saves via the `WriteFile`/`FileWritten` JSON-LD messages.
- **Device selection** (`pick_device`): `AUTO` prefers any `GPU*` device (Iris Xe
  enumerates as `GPU`) and falls back to `CPU`; explicit names are validated against
  `openvino.Core().available_devices`. A GPU load failure in `build_pipeline` auto-retries
  on CPU.
- **GGUF path**: the base OpenVINO runtime cannot read GGUF. The script relies on
  **OpenVINO GenAI** `LLMPipeline(gguf_path, device)`, which natively ingests GGUF for a
  limited set of architectures (llama, qwen2, ...). Unsupported architectures fall back to
  the **llama.cpp engine** (`llama_engine.py`), selected per-request via `Load`'s `engine`.
- **OpenVINO IR path**: `LLMPipeline` also loads an **IR directory** (`openvino_model.xml`
  + `.bin` + tokenizer) with **mmap** (`ov::enable_mmap`, default), so weights page from the
  `.bin`. `es_modelo_ir(path)` detects it; the picker lists local IR dirs (`scan_local_ir`),
  `crear_motor` loads them OpenVINO-only (no planner/llama.cpp), `_contexto_modelo` reads
  `config.json` `max_position_embeddings`, and `-m` accepts an IR dir. `convert_to_ir`
  (`--convert-ir HF_ID --weight-format int4`) shells out to `optimum-cli export openvino`.
  Note: bartowski ships **GGUF**, not IR — IR comes from the `OpenVINO` HF org or this
  converter. mmap helps CPU paging/load; GPU still needs the model to fit shared memory.
- **IR inspection**: `ir_reader.py` (`describe_ir`) reports an IR dir's files+sizes,
  `config.json` (with Spanish meanings), `generation_config.json`, and the XML `rt_info`
  (optimum version, nncf weight compression) — read from the file tail, parsed defensively
  (`defusedxml` if present, else DOCTYPE/ENTITY rejected). Surfaced via `/ir` (rich tables
  in the TUI; `IrHeader`/`IrInfo` JSON-LD for Ink). Tests in `tests/test_ir_reader.py`.
- **IR registry entries** (`MODELS` with `format:"ir"`, `dir`+`repo`): OpenVINO-org IR repos
  (`phi-3-medium-ir`, `mistral-7b-ir`, `phi-3.5-mini-ir`, `tinyllama-ir`) downloaded as a
  full snapshot (`_descargar_ir` → `snapshot_download`) into a directory.
- **Engines**: `serve_stdio` keeps `state["engine"]` (`"openvino"` default or
  `"llamacpp"`); `Generate` dispatches to the active one. Both stream `Token` events.
- **Cancelación / control del terminal**: el subproceso de `LlamaCppEngine` se lanza con
  `stdin=subprocess.DEVNULL` (si no, robaba el stdin del terminal y peleaba con
  prompt_toolkit: caracteres al mover el ratón, cuelgues, Ctrl-C muerto). Cada motor
  expone `cancelar()` (llama.cpp termina el subproceso; llama-server corta el SSE vía
  bandera; OpenVINO se detiene cuando su streamer devuelve True, vía el parámetro
  `debe_parar` de `_motor_generar`). En la `ChatTUI`, `Ctrl-C`/`Esc` durante la generación
  cancelan (recuperan el control) y una 2ª pulsación de `Ctrl-C` sale; el ratón está
  desactivado (`mouse_support=False`) y el scroll de la salida se hace con `RePág`/`AvPág`.

## Implementation notes

- Keep the entry point a single script per the spec, unless the user asks to expand into a
  package.
- There is no test suite or linter configured yet; `py_compile` is the only check.

## Environment

- Platform: Windows 11, PowerShell. Use PowerShell syntax (`$env:VAR`, `$null`) for shell
  commands; Python is cross-platform.
