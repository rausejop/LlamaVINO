---
name: tui-chat-prompt-toolkit
description: Construir una TUI de chat a pantalla completa estilo Claude Code con prompt_toolkit (salida con scroll, barras fijas, streaming desde un hilo, autocompletado de comandos /, cancelación con Ctrl-C). Úsalo al diseñar o depurar la interfaz interactiva de un agente LLM en terminal.
---

# TUI de chat a pantalla completa con prompt_toolkit

Patrón verificado en `LlamaVino.py` (`ChatTUI`). Una `Application` full-screen con
streaming en tiempo real y cancelación suave.

## Layout

`HSplit` de arriba a abajo: ventana de **salida** con scroll (read-only `Buffer` +
`BufferControl(focusable=False)`), ventana opcional de **candidatas**, y FIJOS abajo:
**barra superior / prompt / barra inferior / estado**. Las barras son
`Window(char="─", style=callable)` → ancho completo y se re-fluyen al redimensionar.
El prompt es `multiline=False` (envuelve visualmente). El cursor de la salida se ancla
al final para que siga la conversación.

## Streaming desde un hilo (lo más delicado)

La generación bloquea, así que va en un executor; **no toques la UI desde el hilo**:

```python
self._loop = asyncio.get_event_loop()
resultado = await self._loop.run_in_executor(None, self._trabajo_generar, ...)

def _streamer(self, pieza):          # corre en el hilo del executor
    self.stream_text += pieza
    self._loop.call_soon_threadsafe(self._tick)   # vuelve al hilo de la UI

def _tick(self):
    self._refrescar(); self.app.invalidate()       # redibuja
```

El contador de tokens del estado se actualiza **en vivo** porque `_streamer` incrementa
y agenda `invalidate()` por token.

## Cancelación suave (recuperar el control)

- Lanza el subproceso con `stdin=subprocess.DEVNULL` (ver [[motor-respaldo-llamacpp]]).
- `threading.Event` `_cancelar`; en el binding de `c-c`/`escape`, si está generando,
  `_cancelar.set()` + `engine.obj.cancelar()` (no salir). 2ª pulsación de `c-c` sale.
- Pasa `debe_parar=self._cancelar.is_set` a la generación; el streamer de OpenVINO
  **devuelve True para detenerse**. Al cancelar, conserva lo ya generado como parcial.

## Gotchas verificados

- **`mouse_support=False`.** Con el ratón activado, en algunos terminales el movimiento
  filtra secuencias de escape al prompt (caracteres basura). Desplaza la salida con
  `RePág`/`AvPág` (`buffer.cursor_up/down(n)`) en su lugar.
- **Ctrl-C tardío en Windows**: tras devolver el control, el `Ctrl-C` puede llegar como
  señal real durante `loop.close()` y reaparece como `KeyboardInterrupt`. Envuelve
  `app.run()` en `try/except KeyboardInterrupt: pass`.
- **No anides Applications.** Para `/model` y `/config`, **sal** de la `ChatTUI`
  (`app.exit()` con una `accion`) y reentra conservando el estado (`exportar_estado()`),
  en vez de abrir un selector dentro del bucle de la app.
- **Salida larga legible**: `/help`, `/gguf`, `/ir`, etc. usan `run_in_terminal` con una
  pausa de ENTER (un "pager") para poder leerla/scrollearla en el terminal normal.
- **Errores no deben romper el event loop**: envuelve `_procesar` en `try/except` y
  reporta el fallo en el chat.

## Autocompletado de comandos `/`

`Completer` + `complete_while_typing=True` + un `Lexer` que pinta los comandos en azul;
el menú se filtra al teclear. `CompletionsMenu` como `Float` sobre el cuerpo.

Relacionado: [[puente-ink-python-jsonld]] para la variante de frontend en Node/Ink.
