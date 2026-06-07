# LlamaVINO

Ejecuta modelos de lenguaje (LLM) en **local** sobre una GPU integrada **Intel Iris Xe**,
leyendo modelos en formato **GGUF** (el de llama.cpp) o **OpenVINO IR**, con la librería
**OpenVINO GenAI** y un respaldo automático a **llama.cpp** cuando OpenVINO no puede con
un modelo. Trae dos interfaces tipo *Claude Code*: una TUI de terminal a pantalla completa
y una UI web en Node + Ink.

> © Rafael Ausejo Prieto, con ayuda de Claude Code.

```
 _     _                         __     __  ___   _   _    ___
| |   | | __ _ _ __ ___   __ _   \ \   / / |_ _| | \ | |  / _ \
| |   | |/ _` | '_ ` _ \ / _` |   \ \ / /   | |  |  \| | | | | |
| |___| | (_| | | | | | | (_| |    \ V /    | |  | |\  | | |_| |
|______|_|\__,_|_| |_| |_|\__,_|    \_/    |___| |_| \_|  \___/
```

Este README es **autocontenido**: explica qué es, cómo se instala, cómo se usa, cómo
funciona por dentro, las dificultades técnicas que costó resolver, y un **tutorial de cada
concepto** para que puedas entenderlo aunque vengas de cero.

---

## Índice

1. [¿Qué es esto y por qué existe?](#1-qué-es-esto-y-por-qué-existe)
2. [Tutorial de conceptos (de cero)](#2-tutorial-de-conceptos-de-cero)
3. [Requisitos y hardware](#3-requisitos-y-hardware)
4. [Instalación](#4-instalación)
5. [Uso](#5-uso)
6. [Comandos del chat](#6-comandos-del-chat)
7. [Cómo funciona por dentro (arquitectura)](#7-cómo-funciona-por-dentro-arquitectura)
8. [Dificultades técnicas encontradas](#8-dificultades-técnicas-encontradas)
9. [Pruebas](#9-pruebas)
10. [Estructura de ficheros](#10-estructura-de-ficheros)
11. [Skills (conocimiento reutilizable)](#11-skills-conocimiento-reutilizable)
12. [Créditos y licencia](#12-créditos-y-licencia)

---

## 1. ¿Qué es esto y por qué existe?

**LlamaVINO** es un *runner* local de LLMs pensado para un portátil normal con **GPU
integrada Intel** (Iris Xe) y **16 GB de RAM compartida**, sin tarjeta gráfica dedicada y
sin depender de la nube.

El nombre es la fusión de las dos tecnologías que combina:

- **Llama** → leer modelos en formato **GGUF**, el contenedor que popularizó
  [llama.cpp](https://github.com/ggml-org/llama.cpp).
- **(Open)VINO** → ejecutar con **OpenVINO**, el motor de inferencia de Intel, que sabe
  aprovechar la GPU Iris Xe.

La motivación: en una iGPU con memoria compartida, *qué* modelo cargas y *cómo* lo cargas
lo es todo. LlamaVINO automatiza esas decisiones (motor, reparto de capas GPU/CPU,
cuantización, caché) y te da una experiencia de chat cómoda en el terminal.

**Qué sabe hacer, en una frase por punto:**

- Cargar GGUF e **IR** y generar texto con *streaming* (la respuesta aparece palabra a
  palabra).
- Elegir **automáticamente** el mejor motor: OpenVINO si la arquitectura es nativa y cabe;
  si no (p. ej. Mistral en GGUF, o un modelo que no entra en VRAM), **llama.cpp**.
- **Repartir capas** entre GPU y CPU en tiempo real según la memoria disponible.
- Inspeccionar la cabecera de un GGUF (`/gguf`) o un IR (`/ir`) sin cargar el modelo.
- Convertir modelos de Hugging Face a **OpenVINO IR int4** (`--convert-ir`).
- Mostrar, con `llama-server`, las **palabras candidatas** de cada token y sus
  probabilidades.
- Gestionar modelos (buscar en HF, descargar, borrar) y registrar su **rendimiento**
  (tok/s, segundos de carga) en SQLite.

---

## 2. Tutorial de conceptos (de cero)

Esta sección explica cada pieza del rompecabezas. Si ya dominas un concepto, sáltalo.

### 2.1. LLM, tokens y contexto

Un **LLM** (*Large Language Model*) predice el siguiente fragmento de texto. No trabaja con
caracteres ni con palabras exactas, sino con **tokens**: trozos de palabra (p. ej. `Hola`,
` mun`, `do`). El modelo tiene un **vocabulario** fijo de tokens.

- **Ventana de contexto**: el número máximo de tokens que el modelo puede "tener en la
  cabeza" a la vez (prompt + respuesta + historial). Si la conversación crece más que la
  ventana, hay que recortar o **resumir** (`/compact`).
- En LlamaVINO, la barra de estado muestra `Contexto: n% usado` e `i / j tokens`, donde
  `i` son los tokens en uso ahora mismo y `j` el máximo de la sesión (la ventana del
  modelo).

### 2.2. Cuantización

Los pesos de un modelo son números (originalmente `float16`/`float32`). **Cuantizar** es
guardarlos con menos bits para que ocupen menos y vayan más rápido, a cambio de algo de
precisión.

- **Q4_K_M** (4 bits, variante "K medium") es el equilibrio recomendado para esta iGPU:
  buena calidad y tamaño contenido. `Q4_0` es más pequeño; `Q8_0`/`Q2` se evitan (uno
  pesa demasiado, el otro pierde demasiada calidad).
- En IR de OpenVINO el equivalente es **int4** (pesos comprimidos con NNCF).

### 2.3. GGUF

**GGUF** es el formato de fichero de llama.cpp: un único `.gguf` con una **cabecera de
metadatos** (arquitectura, nombre, vocabulario, parámetros de tokenizador…) seguida de los
**tensores** (los pesos). LlamaVINO sabe leer esa cabecera **sin dependencias** y sin
cargar los pesos (módulo `gguf_reader.py`), lo que permite, en milisegundos, saber qué
arquitectura es un modelo de varios GB y decidir cómo cargarlo.

### 2.4. OpenVINO y OpenVINO GenAI

**OpenVINO** es el motor de inferencia de Intel para CPU/GPU/NPU. El runtime base **no lee
GGUF**. Quien sí lo hace es **OpenVINO GenAI** (`openvino_genai.LLMPipeline`), una capa de
alto nivel que ingiere GGUF de forma nativa para un conjunto de arquitecturas
(`llama`, `qwen2`, `phi3`, `gemma`) y también carga directorios **IR**.

```python
import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline("modelo-Q4_K_M.gguf", "GPU")   # GGUF directo en GPU
```

### 2.5. OpenVINO IR

El **IR** (*Intermediate Representation*) es el formato nativo de OpenVINO: un **directorio**
con `openvino_model.xml` (el grafo), `openvino_model.bin` (los pesos) y el tokenizador. Se
carga con la **misma** `LLMPipeline`, con **mmap** activado, así que los pesos se *paginan*
desde el `.bin` en lugar de copiarse enteros a RAM.

- Es a menudo **más robusto** que el GGUF para arquitecturas que el lector GGUF de OpenVINO
  no digiere (p. ej. Mistral).
- Se obtiene de la organización `OpenVINO` en Hugging Face o convirtiéndolo tú:
  `--convert-ir HF_ID` (usa `optimum-cli export openvino --weight-format int4`).

### 2.6. mmap (memory-mapping)

`mmap` mapea un fichero a memoria virtual y carga sus páginas **bajo demanda**. Permite
*arrancar* modelos más grandes que la RAM (se paginan desde disco; va lento, pero arranca).
llama.cpp lo usa por defecto; el IR de OpenVINO también. Importante: en una iGPU, mmap
ayuda a la carga/paginación en CPU, pero para correr **en GPU** el modelo aún tiene que
caber en la memoria compartida.

### 2.7. Reparto de capas GPU/CPU (`-ngl`)

Un LLM se compone de N **capas** (transformer blocks). Si el modelo entero no cabe en la
"VRAM" (que en la Iris Xe es RAM compartida), se pueden poner **algunas capas en la GPU** y
el resto en CPU/RAM. El parámetro de llama.cpp es `-ngl N` (*number of GPU layers*).

`layer_planner.py` calcula `N` en tiempo real: lee el tamaño del GGUF y su número de capas,
estima el coste de peso por capa **más** el coste del KV-cache por capa (que depende del
contexto), sondea la VRAM (`GPU_DEVICE_TOTAL_MEM_SIZE`) y la RAM libre, y reparte dejando
una reserva para el sistema. Lo ves con `/gpu`.

### 2.8. KV-cache

Durante la generación, el modelo guarda las claves/valores (K/V) de la atención de los
tokens ya vistos para no recalcularlos: es el **KV-cache**. Crece con el contexto. En GPU,
OpenVINO comprime el KV-cache a `u8` por defecto, pero eso **rompe** el chat con historial
(ver [dificultades](#8-dificultades-técnicas-encontradas)); LlamaVINO lo desactiva poniendo
`KV_CACHE_PRECISION=f16` en GPU.

### 2.9. Plantillas de chat

Cada familia de modelos espera el diálogo con un **formato de marcas** concreto:

| Familia | Formato |
| --- | --- |
| Mistral | `[INST] … [/INST]` |
| Gemma | `<start_of_turn>user … <end_of_turn>` |
| Phi-3 | `<|user|> … <|end|><|assistant|>` |
| Qwen | ChatML `<|im_start|>rol … <|im_end|>` |
| genérico | `Usuario: … Asistente:` |

OpenVINO GenAI aplica la plantilla del modelo solo; en el respaldo llama.cpp hay que
construir el prompt a mano. `detect_chat_format()` deduce la plantilla leyendo
`general.name`/`general.architecture` del GGUF.

### 2.10. Streaming, candidatas y `n_probs`

**Streaming** = recibir la respuesta token a token según se genera (no esperar al final).

Con el motor `llama-server` se puede pedir `n_probs`: por cada token, el servidor devuelve
las **k palabras candidatas** y su **probabilidad**. LlamaVINO lo muestra en un panel (la
elegida en verde con `→`, el resto atenuadas) y, en vivo, el token elegido en la barra de
estado. Además guarda un **histórico por token** revisable con `/candidatas historico`.

### 2.11. Identidad del agente

LlamaVINO inyecta un *system prompt* que fija su identidad: "eres **LlamaVINO** (GGUF/
llama.cpp + OpenVINO), no Claude ni el modelo base; solo si te preguntan por el *modelo
fundacional* responde con el real". La temperatura por defecto es **0** (respuestas
deterministas).

### 2.12. La TUI (prompt_toolkit) y el protocolo JSON-LD

La interfaz de terminal es una `Application` de **prompt_toolkit** a pantalla completa:
salida con scroll arriba, y fijos abajo barra/prompt/barra/estado. La generación corre en
un **hilo** (executor) y refresca la UI de forma segura.

La UI web (Node + Ink) es un proceso aparte que lanza `python LlamaVino.py --serve` y habla
con él por un **protocolo JSON-LD por líneas** sobre stdio: una línea JSON por mensaje en
stdout, y todos los logs por stderr para no corromper el stream.

### 2.13. MCP (Model Context Protocol)

**MCP** es un protocolo (JSON-RPC 2.0) para exponer *herramientas* y *recursos* a un agente.
LlamaVINO incluye un cliente mínimo y un servidor de ejemplo (sistema de ficheros confinado
a una raíz), sin SDK externo. Se listan con `/mcp`.

---

## 3. Requisitos y hardware

- **SO**: Windows 11 (probado), PowerShell. El código Python es multiplataforma.
- **Python**: 3.10+ (verificado en 3.14).
- **Hardware de referencia**: Intel i7-1165G7, **Iris Xe** (sin VRAM dedicada: comparte la
  RAM), **16 GB** RAM. Cuenta con que Windows ya usa ~5-6 GB.
- **OpenVINO GenAI** ≥ 2025.1 (verificado con 2026.x).
- Modelos ≤ ~14B en Q4_K_M/int4 para caber en 16 GB compartidos.

> En esta iGPU, un modelo de 1B responde en ~1-3 s; un 7-14B va a unos pocos tok/s. Es
> usable, no instantáneo.

---

## 4. Instalación

```powershell
# 1) Dependencias de Python (OpenVINO GenAI, prompt_toolkit, huggingface_hub)
python -m pip install -r requirements.txt

# 2) (opcional) Frontend web en Node + Ink
cd ui; npm install
```

**Opcionales** (según lo que vayas a usar):

| Para… | Instala |
| --- | --- |
| Convertir HF → IR int4 (`--convert-ir`) | `pip install "optimum-intel[openvino]"` |
| Trazas OpenTelemetry en `--serve` | `pip install "opentelemetry-sdk>=1.27" "opentelemetry-semantic-conventions>=0.48b0"` |
| Análisis de código (CodeOutline/ExtractSymbol) | `pip install tree-sitter tree-sitter-python` |
| Parseo XML endurecido para `/ir` | `pip install defusedxml` |

**Binario de llama.cpp** (para el respaldo y el panel de candidatas): coloca los
ejecutables en `vendor/llama.cpp/` o apunta con `LLAMA_CPP_BIN` / `LLAMA_SERVER_BIN`.
LlamaVINO prefiere **`llama-completion`** sobre `llama-cli` (ver
[dificultades](#8-dificultades-técnicas-encontradas)).

**Hugging Face**: define `HF_TOKEN` para descargas más rápidas/privadas.

---

## 5. Uso

### 5.1. Modos de ejecución

```powershell
python LlamaVino.py                                  # TUI interactiva (recomendado)
python LlamaVino.py --list-models                    # ver el registro de modelos
python LlamaVino.py --list-devices                   # enumerar dispositivos OpenVINO
python LlamaVino.py -m models\gguf\modelo.gguf -p "Hola"   # one-shot desde un GGUF
python LlamaVino.py -m models\Qwen2.5-7B-int4-ov -p "Hola" # one-shot desde un IR
python LlamaVino.py --serve                          # backend JSON-LD (lo usa la UI Ink)
cd ui; npm start                                     # UI web en Ink
```

Los modelos se guardan/buscan en **`models/`** (`models/gguf` y `models/ir`); cámbialo con
`--models-dir`.

### 5.2. Receta recomendada (Iris Xe): GPU + IR int4

OpenVINO IR carga con mmap y la GPU va fina con int4:

```powershell
python LlamaVino.py --convert-ir Qwen/Qwen2.5-14B-Instruct   # → models\...-int4-ov
python LlamaVino.py -m models\Qwen2.5-14B-Instruct-int4-ov
```

### 5.3. Ver las palabras candidatas por token

```powershell
python LlamaVino.py --engine llamaserver -m models\gguf\Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
```

Verás el panel de candidatas y el token elegido en la barra de estado. Alterna la vista con
`/candidatas on|off` y revisa el histórico con `/candidatas historico`.

### 5.4. Flags principales

| Flag | Descripción |
| --- | --- |
| `-m, --model RUTA` | Fichero `.gguf` o directorio IR a cargar. |
| `-p, --prompt TEXTO` | Genera una vez y sale (one-shot). |
| `-d, --device AUTO\|GPU\|CPU` | Dispositivo OpenVINO (AUTO prefiere GPU). |
| `--engine auto\|openvino\|llamacpp\|llamaserver` | Motor (por defecto `auto`). |
| `--n-gpu-layers auto\|N` | Capas a la GPU (`auto` = plan en tiempo real). |
| `--n-ctx N` | Tamaño de contexto. |
| `--n-probs N` | Nº de candidatas por token (motor `llamaserver`, def. 5). |
| `--max-new-tokens N` | Máximo de tokens a generar. |
| `--temperature/--top-p/--top-k` | Muestreo (temperatura por defecto **0.0**). |
| `--presence-penalty/--frequency-penalty` | Penalizaciones de repetición. |
| `--stop TEXTO` | Secuencia(s) de parada. |
| `--no-stream` | Desactiva el streaming. |
| `--download ALIAS` | Descarga un modelo del registro desde HF. |
| `--convert-ir HF_ID [--weight-format int4\|int8\|fp16]` | Convierte HF → IR. |
| `--models-dir RUTA` | Carpeta de modelos (def. `models/`). |
| `--list-models / --list-devices / --version` | Utilidades. |

En GPU se desactiva la compresión del KV-cache (`KV_CACHE_PRECISION=f16`) y se cachea la
compilación (`CACHE_DIR=models/.ovcache`) para recargas rápidas.

---

## 6. Comandos del chat

Al teclear `/` aparece un menú con autocompletado (flechas para elegir, Tab para completar).
Los comandos informativos largos se muestran en un *pager* (ENTER para volver al chat).

| Comando | Qué hace |
| --- | --- |
| `/help` | Lista de comandos. |
| `/clear` | Reinicia la conversación (borra el contexto). |
| `/compact` | Resume la conversación para ahorrar contexto. |
| `/config` | Ajusta temperatura, top-p/k, penalizaciones, tokens, parada. |
| `/model` | Cambia de modelo (selector con flechas; conserva la conversación). |
| `/save [ruta]` | Guarda el último bloque de código generado. |
| `/cost` | Uso de tokens y tiempos de la sesión. |
| `/status` | Modelo, dispositivo, motor y ajustes activos. |
| `/gguf [filtro]` · `/gguf tokens [N]` | Cabecera del GGUF; vuelca arrays. |
| `/ir` | Info del IR cargado (ficheros, config, rt_info). |
| `/gpu` | Reparto de capas GPU/CPU calculado en tiempo real. |
| `/color [color]` | Color de las barras del prompt. |
| `/doctor` | Diagnóstico de OpenVINO y dispositivos. |
| `/mcp` | Lista los servidores MCP configurados. |
| `/candidatas [on\|off\|historico]` | Vista en vivo de candidatas / histórico. |
| `/exit` | Sale. |

**Control durante la generación**: `Ctrl-C` (o `Esc`) **cancela** y recupera el control
(conserva lo ya generado); una 2ª pulsación de `Ctrl-C` sale. El scroll de la salida es con
`RePág`/`AvPág` (el ratón está desactivado a propósito).

---

## 7. Cómo funciona por dentro (arquitectura)

### 7.1. Visión general

```
            ┌───────────────────────┐         ┌──────────────────────────────┐
            │  TUI rich/prompt_toolkit │  o  │  UI Node + Ink (ui/)         │
            └───────────┬───────────┘         └──────────────┬───────────────┘
                        │ (mismo proceso)        JSON-LD por stdio (líneas)
                        │                                      │
                  ┌─────▼──────────────────────────────────────▼─────┐
                  │                 LlamaVino.py                       │
                  │   crear_motor()  ·  _motor_generar()  ·  serve     │
                  └───┬───────────────┬───────────────┬───────────────┘
                      │               │               │
              ┌───────▼──┐    ┌───────▼───────┐  ┌────▼─────────────┐
              │ OpenVINO │    │ llama.cpp CLI │  │ llama-server     │
              │  GenAI   │    │ (subproceso)  │  │ (HTTP/SSE,n_probs)│
              └──────────┘    └───────────────┘  └──────────────────┘
                      │               │               │
              GGUF nativo / IR   GGUF vía binario   GGUF + candidatas
```

### 7.2. Selección de motor (`crear_motor`)

`--engine auto` (por defecto):

1. Lee la **arquitectura** del GGUF (cabecera) o detecta un IR.
2. Si la arquitectura es nativa de OpenVINO (`llama/qwen2/phi3/gemma`) **y** cabe en VRAM →
   **OpenVINO GenAI**.
3. Si no cabe entera, o la arquitectura no es nativa (p. ej. Mistral), o OpenVINO falla al
   leer el GGUF → **llama.cpp** (con mmap y `-ngl` del plan), **si hay binario**; si no, un
   mensaje claro pidiendo instalarlo o elegir otro modelo.
4. `llamaserver` solo si lo pides explícitamente (es quien expone las candidatas).

### 7.3. Generación unificada (`_motor_generar`)

Una sola función despacha la generación a los tres motores y unifica el *streaming*, el
*system prompt* de identidad, el contador de tokens y la **cancelación** (`debe_parar()`):
en OpenVINO el *streamer* devuelve `True` para parar; en llama.cpp/server se llama a
`engine.cancelar()`.

### 7.4. El protocolo JSON-LD (`--serve`)

Cada mensaje lleva `@context` (de `context.jsonld`) y `@type`. Ejemplos: `Load`→`Loaded`,
`Generate`→ varios `Token`, `WriteFile`→`FileWritten`, `GgufHeader`/`GgufInfo`,
`IrHeader`/`IrInfo`, `CodeOutline`, `ExtractSymbol`. **Regla de oro**: el protocolo va por
stdout; **todo** log va por stderr. Las rutas pasan por `_confined_path` (confinadas al
workspace).

### 7.5. Persistencia y rendimiento

- Ajustes de la sesión (settings, color, último modelo, `candidatas_vivo`) →
  `.llamavino.json`.
- Rendimiento por modelo (segundos de carga, tok/s) → SQLite `models/llamavino.db`
  (`perf_db.py`), mostrado en la columna **Rend.** del selector.

---

## 8. Dificultades técnicas encontradas

Esta es la parte más útil para quien quiera entender *por qué* el código es como es. Cada
punto costó diagnosticarlo.

### 8.1. `get_state API is supported only when KV-cache compression is disabled` (GPU)

**Síntoma**: con un modelo en GPU, el chat con historial petaba al segundo turno.
**Causa**: el chat reconstruye el estado con `get_state`, incompatible con la compresión
`u8` del KV-cache que OpenVINO activa por defecto en GPU.
**Solución**: poner `KV_CACHE_PRECISION=f16` en la config del plugin **solo en GPU**.

### 8.2. Mistral: `invalid map<K,T> key` al leer el GGUF con OpenVINO

**Síntoma**: ciertos GGUF (Mistral v0.3) no cargaban en OpenVINO GenAI.
**Causa**: metadatos/arquitectura que el lector GGUF de OpenVINO no ingiere.
**Solución**: bajo `--engine auto`, detectar el caso y **caer a llama.cpp** automáticamente;
o usar la versión **IR** del modelo (más robusta).

### 8.3. El gran cuelgue de Mistral: `llama-cli` se volvió interactivo

**Síntoma**: con Mistral (que va por llama.cpp) la interfaz "se quedaba pillada", aparecían
caracteres al mover el ratón y **ni Ctrl-C ni Ctrl-Z** respondían.
**Diagnóstico**: dos problemas encadenados.

1. **Contención de stdin**: el subproceso de llama.cpp se lanzaba sin `stdin=`, así que
   **heredaba el stdin del terminal** y peleaba con prompt_toolkit (de ahí los caracteres
   del ratón y el Ctrl-C muerto). → **Solución**: `stdin=subprocess.DEVNULL`.
2. **`llama-cli` ya no es one-shot**: los builds recientes de llama.cpp (b9xxx+) convirtieron
   `llama-cli` en una herramienta **siempre interactiva**, que **rechaza `-no-cnv`**
   (*"please use llama-completion instead"*) y se queda esperando entrada por stdin → cuelgue.
   → **Solución**: preferir el binario **`llama-completion`** (no interactivo); en builds
   antiguos sin él, recaer en `llama-cli`/`main`.

Como `llama-completion` emite **códigos ANSI de color** en stdout, se añadió un filtro por
fragmentos (`_crear_filtro_ansi`, conserva secuencias partidas en el borde) y se recorta el
marcador final `[end of text]`.

### 8.4. Recuperar el control sin matar el proceso

**Requisito**: poder cancelar una generación "a lo suave". **Solución**: cada motor expone
`cancelar()` (llama.cpp termina el subproceso; llama-server corta el SSE con una bandera;
OpenVINO se detiene cuando su *streamer* devuelve `True`). En la TUI, `Ctrl-C`/`Esc` durante
la generación cancelan; una 2ª pulsación sale. Y se desactivó el ratón
(`mouse_support=False`) porque filtraba escapes al prompt.

### 8.5. `Ctrl-C` tardío en Windows = traceback feo

**Síntoma**: al salir con Ctrl-C, un `KeyboardInterrupt` durante `loop.close()`.
**Causa**: tras devolver prompt_toolkit el control y restaurar el terminal, el Ctrl-C llega
como señal real durante el cierre del bucle asyncio.
**Solución**: envolver `app.run()` en `try/except KeyboardInterrupt: pass`.

### 8.6. El `/ir` se mostraba y desaparecía (sin scroll)

**Síntoma**: la salida larga de `/ir` no daba tiempo a leerla y desactivaba el scroll.
**Solución**: un *pager* (`run_in_terminal` + pausa de ENTER) para los comandos
informativos largos.

### 8.7. "No veo las candidatas"

**Síntoma**: el panel de candidatas no aparecía o pasaba volando. **Causas**: (a) solo
existen con `--engine llamaserver` — con otros motores no se generan; (b) el panel se ocultaba
al terminar la generación y, como cambian por token, no daba tiempo. **Solución**: el panel
**persiste tras generar** (muestra el último token), hay un **histórico** revisable
(`/candidatas historico`) y se puede alternar la vista en vivo (`/candidatas on|off`).

### 8.8. El banner ponía "Llam VINO"

Un *off-by-one* en el ASCII art comía la `a`. Se regeneró el banner para que diga
**LlamaVINO**.

### 8.9. Codepage de Windows y acentos rotos (`�`)

No es el modelo: es la codepage de la consola. Se reconfigura `stdout`/`stderr` a UTF-8 al
arrancar.

---

## 9. Pruebas

```powershell
python -m py_compile LlamaVino.py llama_engine.py    # comprobación de sintaxis
python -m unittest discover tests                    # suite de tests (~100)
cd ui; node --check source/app.js                    # sintaxis de la UI Ink
cd ui; node test-backend.mjs                          # prueba el puente Node↔Python (sin TTY)
```

> Nota: en máquinas con un binario de llama.cpp instalado, el test `test_missing_raises`
> falla a propósito (comprueba la ausencia del binario); es dependiente del entorno, no una
> regresión.

---

## 10. Estructura de ficheros

| Fichero/módulo | Responsabilidad |
| --- | --- |
| `LlamaVino.py` | CLI + motor + TUI rich + backend `--serve`. |
| `llama_engine.py` | Motores de respaldo llama.cpp (`LlamaCppEngine`, `LlamaServerEngine`). |
| `layer_planner.py` | Plan de reparto de capas GPU/CPU (`-ngl`) en tiempo real. |
| `gguf_reader.py` | Lectura de la cabecera GGUF sin dependencias. |
| `ir_reader.py` | Inspección de un directorio IR (`describe_ir`). |
| `perf_db.py` | SQLite de rendimiento por modelo. |
| `edit_formats.py` | Aplicadores de parches (diff unified + aider) y escritura segura en Windows. |
| `code_structure.py` | Análisis de código con tree-sitter. |
| `mcp_client.py`, `mcp_servers/` | Cliente y servidor MCP mínimos (JSON-RPC 2.0). |
| `ui/` | Frontend Node + Ink (ver `ui/README.md`). |
| `context.jsonld` | Vocabulario JSON-LD del protocolo. |
| `system_prompt.md` | Núcleo del *system prompt* de seguridad/identidad. |
| `skills/` | Conocimiento reutilizable en formato Agent Skills. |
| `tests/` | Suite de pruebas. |
| `CLAUDE.md` | Guía detallada de arquitectura (para Claude Code y humanos). |
| `ESTANDARES.md` | Especificación de estándares (JSON-LD, OTel, MCP, LSP…). |

---

## 11. Skills (conocimiento reutilizable)

En `skills/` hay *Agent Skills* (un `SKILL.md` por carpeta) que capturan el conocimiento con
gotchas verificados de este proyecto, listos para reutilizar:

- `ejecutar-gguf-en-iris-xe` — ruta nativa OpenVINO en Iris Xe.
- `motor-respaldo-llamacpp` — el motor llama.cpp y el arreglo `llama-completion`.
- `tui-chat-prompt-toolkit` — TUI a pantalla completa con streaming y cancelación.
- `puente-ink-python-jsonld` — protocolo JSON-LD Node↔Python por stdio.
- `leer-cabecera-gguf` — parsear la cabecera GGUF sin dependencias.
- `openvino-ir-cargar-convertir` — cargar/convertir/inspeccionar IR.
- `reparto-capas-gpu-cpu` — planificar `-ngl` GPU vs RAM.
- `escritura-y-parches-windows` — escritura segura en Windows + diffs.
- `candidatos-por-token-llama-server` — candidatas por token con `n_probs`.

---

## 12. Créditos y licencia

**© Rafael Ausejo Prieto, con ayuda de Claude Code.**

Proyecto en español (UI, comentarios y docstrings). JSON es **JSON-LD**; Markdown es
**CommonMark**; Python sigue PEP 8/257/484 + Google Python Style Guide.

Para el detalle de arquitectura y protocolo, ver [`CLAUDE.md`](CLAUDE.md); para la UI web,
[`ui/README.md`](ui/README.md); para los estándares, [`ESTANDARES.md`](ESTANDARES.md).
