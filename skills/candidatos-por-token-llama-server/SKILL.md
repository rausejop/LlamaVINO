---
name: candidatos-por-token-llama-server
description: Exponer las palabras candidatas y sus probabilidades por cada token generado, lanzando llama-server y leyendo /completion con n_probs por SSE. Úsalo para mostrar un panel de candidatos por token o inspeccionar la distribución del muestreo.
---

# Candidatos por token con llama-server (n_probs, SSE)

Verificado en `LlamaServerEngine` (`llama_engine.py`). Ni OpenVINO GenAI ni `llama-cli`
exponen las probabilidades por token; **`llama-server`** sí, vía `/completion` con
`n_probs`.

## Cómo

1. Localiza el binario `llama-server` (`LLAMA_SERVER_BIN` o `vendor/llama.cpp/`) y
   arráncalo en un **puerto libre** (bind a `127.0.0.1:0` para que el SO asigne uno).
2. Espera a que el endpoint responda antes de generar (poll a `/health` o reintentos).
3. `POST /completion` con `stream:true` y `n_probs:N` (p. ej. 5). Lee la respuesta como
   **SSE** (`data: {…}` por línea); cada token llega con sus top-k candidatos.

```python
payload = {"prompt": prompt, "n_predict": max_new, "stream": True, "n_probs": 5,
           "temperature": temp, "top_p": top_p, "top_k": top_k}
```

## Normalizar las probabilidades (formatos cambiantes)

llama.cpp ha cambiado el esquema entre versiones. Tolera todas:
`probs` / `top_logprobs`, con `tok_str` / `token` / `content` para la palabra y
`prob` / `logprob` para el valor. Un `_normaliza_candidatos` que pruebe esas claves evita
romperse al actualizar el binario.

## Render (panel + barra de estado + histórico)

Por token: muestra la palabra **elegida en verde con `→`** y el resto atenuadas, encima
del texto que se va escribiendo (ventana de candidatas sobre la salida, ver
[[tui-chat-prompt-toolkit]]). El **token elegido** también se muestra en vivo en la barra
de estado, entre el contexto y los tokens.

- **Toggle de vista en vivo**: comando `/candidatas [on|off]` (bandera `candidatas_vivo`,
  persistida y por defecto **on**). Apaga el panel y el indicador de la barra, pero…
- **Histórico siempre**: registra cada token (`{pos, elegida, lista}`) en `_on_cands`
  aunque la vista esté apagada; se limpia al empezar cada generación. `/candidatas
  historico` lo vuelca en un pager (una línea por token: elegida + alternativas).

## Gotchas

- **Solo aparecen con `--engine llamaserver`.** Con `auto`/`openvino`/`llamacpp` no se
  generan candidatas y el panel nunca se muestra (confusión típica: "no las veo"). Lánzalo
  así: `python LlamaVino.py --engine llamaserver -m models/gguf/MODELO.gguf`.
- **Persiste el panel tras generar.** Si el filtro de visibilidad exige `generando`, las
  candidatas desaparecen al acabar y, como cambian en cada token, pasan volando sin que se
  puedan leer. Deja el panel visible mientras `bool(lista)` y límpialo al **empezar** la
  siguiente generación (no al terminar).
- **`stdin=subprocess.DEVNULL`** al lanzar el servidor (igual que el CLI: no robar el
  stdin del terminal). Ver [[motor-respaldo-llamacpp]].
- **Cancelación**: corta el bucle de lectura del SSE con una bandera (`self._cancelar`);
  no hace falta matar el servidor en cada cancelación, pero sí pararlo al salir/cambiar
  de modelo (`cerrar_motor`).
- Solo disponible en este motor (`--engine llamaserver`); los demás no dan probabilidades.
