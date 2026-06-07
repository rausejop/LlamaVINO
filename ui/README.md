# LlamaVino · Interfaz Ink

Interfaz de terminal estilo Claude Code para LlamaVino, construida con
[Ink](https://github.com/vadimdemedes/ink) (React + Yoga Layout + Chalk). Es el
**frontend**; el motor de inferencia es Python (`../LlamaVino.py --serve`), con el
que se comunica por un protocolo **JSON-LD** por líneas sobre stdin/stdout. Toda la
interfaz está en español.

## Requisitos

- Node.js 18+ (probado con v24).
- El backend Python ya instalado: `python -m pip install -r ../requirements.txt`.

## Uso

```powershell
npm install      # una sola vez
npm start        # lanza la interfaz
```

Variable opcional: `LLAMAVINO_PYTHON` para apuntar a un intérprete concreto
(por ejemplo, el de una venv).

## Controles

- **Hub de ajustes:** `↑`/`↓` para moverte, `←`/`→` para cambiar un valor, `Enter`
  para activar, `q` para salir. La barra inferior muestra **siempre** los valores.
- **Selector de modelos:** `↑`/`↓`, `Enter` para elegir/descargar, `q` para volver.
- **Chat:** escribe y `Enter`. `Esc` vuelve al hub. Comandos compatibles con
  Claude Code (adaptados al motor local): `/help`, `/clear`, `/compact`,
  `/config`, `/model`, `/save`, `/cost`, `/status`, `/gguf`, `/doctor`, `/mcp`,
  `/menu`, `/exit`. Atajos rápidos extra: `/temp`, `/top_p`, `/top_k`, `/max`,
  `/stream`.
- **`/gguf`:** muestra la cabecera del modelo cargado (parámetros + significado).
  `/gguf <filtro>` filtra por clave (p. ej. `/gguf rope`); si el filtro coincide
  con un array (`/gguf tokens`) vuelca su contenido completo.
- **Menú de comandos:** al teclear `/` aparece la lista de comandos; se filtra
  por prefijo conforme escribes, `↑`/`↓` selecciona, `Tab` completa, `Enter`
  ejecuta el resaltado y `Esc` cierra el menú.
- **Pantalla `/configuration`:** `↑`/`↓` para moverte, `←`/`→` para ajustar
  numéricos, `Enter` sobre «Secuencias de parada» para escribirlas, `q` para volver.

## Estructura

| Archivo | Función |
| --- | --- |
| `source/cli.js` | Punto de entrada; lanza el backend y monta la app. |
| `source/backend.js` | Puente JSON-LD con Python (spawn seguro, sin shell). |
| `source/app.js` | Máquina de estados y lógica (React). |
| `source/components.js` | Componentes visuales (cabecera, barra de estado, chat). |
| `test-backend.mjs` | Prueba del puente sin TTY. |

El protocolo y el resto de estándares están en [`../ESTANDARES.md`](../ESTANDARES.md).
