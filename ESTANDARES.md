# Estándares de LlamaVino

Este documento define los estándares que adopta el proyecto. Sigue la
especificación [CommonMark](https://spec.commonmark.org/). Cada estándar indica
su **estado**: `Implementado`, `Parcial` o `Roadmap`, y su **punto de integración**
en el código.

El proyecto se construye **por capas**: el núcleo verificado funciona hoy; el resto
queda definido aquí con su interfaz, para añadirlo sin reescribir lo existente.

## Resumen de estado

| Área | Estándar | Estado |
| --- | --- | --- |
| Motor principal | OpenVINO GenAI (GGUF en GPU) | Implementado |
| Motor respaldo | llama.cpp (binario, mmap/`-ngl`) | Implementado (requiere binario) |
| Interfaz | Ink (React/Yoga) + Chalk | Implementado |
| Protocolo | JSON-LD | Implementado |
| Vocabulario | Schema.org | Parcial (`context.jsonld`) |
| Mensajes | Roles OpenAI/Anthropic (system/user/assistant/tool) | Implementado |
| Observabilidad | OpenTelemetry GenAI | Implementado (consola) |
| Markdown | CommonMark + subconjunto seguro (CSP) | Parcial |
| Markdown UI | MDX | Roadmap |
| Herramientas | MCP | Implementado (interfaz + servidor ejemplo) |
| IDE | LSP | Roadmap |
| Edición | Git Unified Diff + bloques Aider | Implementado |
| Código | Tree-sitter | Implementado (Python) |
| Procedencia | W3C Verifiable Credentials | Roadmap |
| Ficheros Windows | Rutas largas `\\?\` + locking con backoff | Implementado |
| Paquetes | Winget (manifiestos) | Roadmap |
| Seguridad Windows | Execution Policy + UAC `requires_elevation` | Roadmap |
| Shell | PowerShell Core (objetos) + Win32 | Parcial |
| Calidad Python | PEP 8/257/484 + Google Style Guide | Implementado |
| Agente | Núcleo con harness de seguridad | Parcial (confinamiento de ficheros) |

---

## 1. Interfaz: Ink + Chalk · Implementado

- **Punto de integración:** `ui/source/`.
- Frontend en Node + React mediante [Ink](https://github.com/vadimdemedes/ink)
  (layout Flexbox con Yoga, estado reactivo con `useState`/`useEffect`, colores con
  Chalk). Sin paso de compilación: se usa `htm` para JSX en plantillas.
- La barra de estado inferior muestra **en todo momento** los valores de todas las
  opciones; se navegan con flechas (↑/↓ ←/→) y también por comandos `/`.

## 2. Protocolo semántico: JSON-LD · Implementado

- **Punto de integración:** `LlamaVino.py` (`_ld`, `serve_stdio`) y
  `ui/source/backend.js`.
- Cada mensaje del protocolo stdin/stdout es un documento JSON-LD con `@context`,
  `@type` y `@id` (correlación petición/respuesta).
- Tipos de petición: `ListModels`, `ListDevices`, `Download`, `Load`, `Generate`,
  `FileActionRequest`, `CodeOutline`, `ExtractSymbol`, `McpListServers`,
  `McpListTools`, `McpCallTool`, `Ping`, `Quit`. Tipos de respuesta: `Ready`,
  `Models`, `Devices`, `Downloaded`, `Loaded`, `Token`, `Done`, `Patched`,
  `Outline`, `Symbol`, `McpServers`, `McpTools`, `McpResult`, `Error`, `Fatal`,
  `Pong`.
- **`FileActionRequest`** (`action: "patch"`) aplica un parche (`patchType:
  GitUnifiedDiff` o `Aider`) usando `edit_formats`, **confinado al workspace**
  (`_confined_path` rechaza rutas con `..` que escapen de la raíz); responde
  `Patched`.

## 3. Vocabulario: Schema.org · Parcial

- **Punto de integración:** `context.jsonld`.
- El contexto mapea los términos del protocolo a tipos de
  [Schema.org](https://schema.org/) (`Message`, `CreativeWork`,
  `SoftwareApplication`) y a un vocabulario propio para lo que no existe (`Prompt`,
  `AIModel`). Roadmap: publicar el contexto en una IRI dereferenciable.

## 4. Estructura de mensajes: roles estándar · Implementado

- **Punto de integración:** campo `history` de `Generate`.
- Se usan los roles de facto de OpenAI/Anthropic: `system`, `user`, `assistant`,
  `tool`. El backend los pasa a `openvino_genai.ChatHistory`. El rol `tool` queda
  reservado para los resultados de MCP (sección 7).

## 5. Observabilidad: OpenTelemetry GenAI · Implementado (consola)

- **Punto de integración:** `LlamaVino.py` (`_init_telemetry`, span `chat`).
- Spans con [convenciones semánticas GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/):
  `gen_ai.system`, `gen_ai.request.model`, `gen_ai.request.max_tokens`,
  `gen_ai.request.temperature`, `gen_ai.usage.input_tokens`,
  `gen_ai.usage.output_tokens`.
- Exportador a **consola/stderr** (sin colector externo). Es **opcional**: si el SDK
  de OTel no está instalado, el agente funciona igual. Para OTLP, basta con definir
  las variables `OTEL_*` estándar.

## 6. Markdown: CommonMark + subconjunto seguro · Parcial

- **Estándar base:** [CommonMark](https://spec.commonmark.org/) para toda la
  documentación.
- **CSP para LLM (sanitización):** el Markdown generado por el modelo debe pasar por
  una allow-list antes de renderizarse, para mitigar *prompt injection* indirecta:
  - **Permitido:** encabezados, énfasis, listas, tablas, citas, bloques de código.
  - **Bloqueado/saneado:** HTML embebido (`<script>`, `<img>`, `<iframe>`…),
    `javascript:`/`data:` en enlaces, imágenes remotas que filtren tokens, y enlaces
    autoclicables. Las URLs se muestran como texto, no se cargan recursos.
- **MDX (Roadmap):** para renderizar componentes interactivos desde el output del
  LLM en una UI web, con la misma allow-list de componentes.

## 7. Herramientas externas: MCP · Implementado (interfaz + servidor ejemplo)

- **Estándar:** [Model Context Protocol](https://modelcontextprotocol.io/)
  (JSON-RPC 2.0 sobre stdio).
- **Punto de integración:** `mcp_client.py` (cliente) y
  `mcp_servers/filesystem_server.py` (servidor de ejemplo). Tests:
  `tests/test_mcp.py` (7 casos), que arrancan el servidor real y hacen el ciclo
  `initialize` → `tools/list` → `tools/call`.
- **Cliente** (`MCPClient`): handshake `initialize` + `notifications/initialized`,
  `list_tools`, `call_tool`; lanza el servidor con argumentos en array (sin shell)
  y es gestor de contexto.
- **Servidor de ejemplo** (filesystem, solo lectura): herramientas `list_dir` y
  `read_file` **confinadas** a la raíz `--root` (rechaza `..`). Sin dependencias
  externas: stdlib pura.
- **Integración con `--serve`:** tipos `McpListServers` / `McpListTools` /
  `McpCallTool`. **Seguridad:** los comandos de servidor salen solo de
  `mcp_config.json` (que controla el usuario); el modelo/protocolo referencia un
  servidor **por nombre**, nunca con un comando arbitrario — coherente con el
  harness (sección 16).
- **Limitación honesta:** un modelo 1B en Iris Xe no realiza *tool-calling* fiable;
  por eso conectar el bucle de herramientas al chat (que el modelo decida qué
  herramienta llamar) y el descubrimiento multi-servidor quedan en Roadmap. El rol
  `tool` del protocolo (sección 4) está reservado para transportar los resultados.

## 8. IDE: Language Server Protocol · Roadmap

- El agente debe consultar un servidor [LSP](https://microsoft.github.io/language-server-protocol/)
  (p. ej. Pyright) para *go-to-definition*, diagnósticos y autocompletado reales, en
  vez de adivinar leyendo texto. Punto de integración previsto: módulo `tools/lsp`.

## 9. Edición de código: diffs · Implementado

- **Punto de integración:** `edit_formats.py`; tests en `tests/test_edit_formats.py`
  (17 casos, incluido round-trip auto-validado con `difflib`).
- **Git Unified Diff** (`apply_unified_diff`): parser tolerante (acepta diffs de
  `git` y de `difflib`), localización de hunks con *fuzz* para tolerar desfases de
  línea, preserva el salto final.
- **Bloques Search & Replace estilo Aider** (`parse_search_replace` /
  `apply_search_replace`): `<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE`, con
  coincidencia tolerante a espacios finales y creación de fichero (SEARCH vacío).
- `apply_patch_to_file(path, patch, fmt=...)` aplica y guarda usando la sección 13.

## 10. Estructura de código: Tree-sitter · Implementado (Python)

- **Punto de integración:** `code_structure.py`; tests en
  `tests/test_code_structure.py` (12 casos). Protocolo: `CodeOutline` y
  `ExtractSymbol` (con confinamiento de directorio).
- Con [Tree-sitter](https://tree-sitter.github.io/) se extraen funciones, clases y
  *scopes* exactos por nombre cualificado (`Clase.metodo`), incluyendo decoradores,
  para enviar al modelo solo la porción relevante en vez de texto plano.
- `list_symbols`, `extract_symbol`, `outline`, `language_for_path`. Soporta Python;
  ampliable registrando más grammars en `_LANGUAGES`.

## 11. Procedencia: W3C Verifiable Credentials · Roadmap

- Firmar criptográficamente outputs del modelo y artefactos con
  [VC](https://www.w3.org/TR/vc-data-model/) para garantizar origen e integridad. El
  documento JSON-LD ya es compatible con la estructura de prueba de las VC.

## 12. Paquetes: Winget · Roadmap

- Toda instalación de herramientas pasa por
  [Winget](https://learn.microsoft.com/windows/package-manager/) con manifiestos
  verificados, en modo silencioso. **Prohibido** descargar `.exe` sueltos de
  internet.

## 13. Ficheros en Windows · Implementado

- **Punto de integración:** `edit_formats.py` (`windows_long_path`,
  `write_text_with_retry`, `read_text`).
- **Rutas largas:** prefijo `\\?\` para rutas absolutas en Windows, superando el
  límite `MAX_PATH` (260).
- **Locking:** `write_text_with_retry` reintenta con *backoff* exponencial ante
  `PermissionError` (fichero abierto por compilador/IDE) antes de rendirse.

## 14. Seguridad de ejecución en Windows · Roadmap

- **Execution Policy:** ejecutar scripts en *scope* de proceso (`RemoteSigned`/
  aislado) sin alterar la política global del sistema ni exigir Administrador.
- **UAC:** el JSON-LD contempla `requires_elevation: true`; al detectarlo, el
  programa lanza el prompt de elevación de Windows de forma segura en vez de
  *crashear*.

## 15. Shell: PowerShell Core + Win32 · Parcial

- El agente estandariza salidas en **PowerShell Core (pwsh)**, basado en objetos,
  para que el JSON-LD parsee resultados estructurados (no texto plano). Nada de
  `.bat`/`cmd.exe`/Bash.
- **Win32 / Windows Terminal:** secuencias de escape y API nativas para control
  avanzado de terminal (Roadmap).

## 16. Núcleo de agente con harness de seguridad · Parcial

Estándar de operación del agente (ver `system_prompt.md`):

- **Hardening de directorios · Implementado:** `_confined_path` en `serve_stdio`
  confina toda acción de fichero a la raíz del workspace y rechaza rutas con `..`
  que escapen (verificado por test end-to-end del protocolo). El resto (abstracción
  de comandos `pwsh`, operaciones prohibidas) queda como Roadmap.
- **Abstracción de comandos:** peticiones como acciones JSON-LD o `pwsh` atómico;
  sin encadenar `&&`/`||`/`;`/`|` salvo pipelines de datos seguros.
- **Operaciones prohibidas:** borrado destructivo fuera del proyecto, manipulación
  del registro, y descarga/ejecución de binarios fuera de Winget.

## 17. Calidad de código Python · Implementado

- [PEP 8](https://peps.python.org/pep-0008/) (estilo),
  [PEP 257](https://peps.python.org/pep-0257/) (docstrings),
  [PEP 484](https://peps.python.org/pep-0484/) (type hints) y la
  [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
  (docstrings con secciones `Args:`/`Returns:`, imports a nivel de módulo, errores
  explícitos).
- **Desarrollo seguro:** sin `eval`/`shell=True`; el frontend lanza Python con
  argumentos en array (sin shell); el protocolo separa datos (stdout) de logs
  (stderr); entradas validadas y errores reportados al UI, nunca silenciados.
