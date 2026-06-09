#!/usr/bin/env python
"""Benchmark de backends de inferencia para LlamaVINO en esta máquina (Iris Xe).

Compara, sobre los **mismos** ficheros GGUF/IR, la velocidad de:

  * **llama.cpp + Vulkan** (GPU)   — `llama-bench` del build actual (`vendor/llama.cpp`).
  * **llama.cpp + CPU**            — el mismo `llama-bench` con `-ngl 0`.
  * **llama.cpp + SYCL** (GPU)     — *opcional*: otro `llama-bench` compilado con oneAPI,
                                     indicado en la variable de entorno ``LLAMA_SYCL_BENCH``.
  * **OpenVINO GenAI** (GPU)       — *opcional* (`--openvino`): carga el modelo con
                                     ``LlamaVino.crear_motor`` y mide tok/s contando tokens.

La pregunta que responde: para el **respaldo llama.cpp**, ¿conviene SYCL en vez de Vulkan
en esta Iris Xe? Y de paso, ¿cómo queda OpenVINO frente a ambos?

Métricas (tokens/segundo, más alto = mejor):
  * **pp** (*prompt processing*): velocidad procesando el prompt (prefill).
  * **tg** (*token generation*): velocidad generando texto nuevo (lo que se «siente»).

Uso::

    python bench/benchmark.py                       # Vulkan vs CPU sobre los GGUF locales
    python bench/benchmark.py --openvino            # añade OpenVINO (solo archs soportadas)
    python bench/benchmark.py -p 256 -n 128 -r 3    # carga de trabajo más larga
    $env:LLAMA_SYCL_BENCH = "C:\\ruta\\sycl\\llama-bench.exe"  # habilita la columna SYCL
    python bench/benchmark.py --openvino

Sin dependencias externas (solo la stdlib); OpenVINO solo si se pide ``--openvino``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Raíz del proyecto (este fichero vive en bench/).
RAIZ = Path(__file__).resolve().parent.parent
BENCH_VULKAN = RAIZ / "vendor" / "llama.cpp" / "llama-bench.exe"
DIR_GGUF = RAIZ / "models" / "gguf"
DIR_RESULTADOS = RAIZ / "bench"


@dataclass
class Medicion:
    """Una fila de resultado: un modelo con un backend, sus tok/s de pp y tg."""

    modelo: str
    backend: str
    pp_ts: float | None = None        # tokens/s procesando el prompt
    tg_ts: float | None = None        # tokens/s generando
    load_s: float | None = None       # segundos de carga (solo OpenVINO)
    error: str | None = None          # motivo si el backend falló para este modelo
    extra: dict = field(default_factory=dict)


def _descubrir_gguf(rutas_cli: list[str]) -> list[Path]:
    """Devuelve los GGUF a medir: los pasados por CLI, o todos los de models/gguf."""
    if rutas_cli:
        return [Path(r) for r in rutas_cli]
    if not DIR_GGUF.is_dir():
        return []
    return sorted(DIR_GGUF.glob("*.gguf"), key=lambda p: p.stat().st_size)


def _correr_llama_bench(binario: Path, modelo: Path, *, ngl: int, n_prompt: int,
                        n_gen: int, reps: int) -> tuple[float | None, float | None, str | None]:
    """Ejecuta ``llama-bench`` y devuelve ``(pp_ts, tg_ts, error)``.

    Parsea la salida JSON: cada objeto es o bien una fila pp (``n_prompt`` > 0) o una
    fila tg (``n_gen`` > 0), con su ``avg_ts`` en tokens/segundo.
    """
    cmd = [
        str(binario), "-m", str(modelo),
        "-ngl", str(ngl),
        "-p", str(n_prompt), "-n", str(n_gen),
        "-r", str(reps),
        "-o", "json",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except FileNotFoundError:
        return None, None, f"binario no encontrado: {binario}"
    except subprocess.TimeoutExpired:
        return None, None, "timeout (>30 min)"
    if proc.returncode != 0:
        cola = (proc.stderr or proc.stdout or "").strip().splitlines()
        return None, None, "; ".join(cola[-2:]) or f"código {proc.returncode}"
    try:
        filas = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None, None, "salida JSON no parseable"

    pp_ts = tg_ts = None
    for fila in filas:
        if int(fila.get("n_prompt", 0)) > 0:
            pp_ts = float(fila.get("avg_ts", 0.0))
        elif int(fila.get("n_gen", 0)) > 0:
            tg_ts = float(fila.get("avg_ts", 0.0))
    return pp_ts, tg_ts, None


def medir_llamacpp(modelos: list[Path], *, n_prompt: int, n_gen: int,
                   reps: int) -> list[Medicion]:
    """Mide Vulkan (GPU, ngl=99), CPU (ngl=0) y, si está, SYCL para cada modelo."""
    sycl_bin = os.environ.get("LLAMA_SYCL_BENCH")
    backends: list[tuple[str, Path, int]] = [
        ("llama.cpp · Vulkan (GPU)", BENCH_VULKAN, 99),
        ("llama.cpp · CPU", BENCH_VULKAN, 0),
    ]
    if sycl_bin:
        backends.append(("llama.cpp · SYCL (GPU)", Path(sycl_bin), 99))
    else:
        print("[i] LLAMA_SYCL_BENCH no definido → se omite la columna SYCL "
              "(ver instrucciones al final).", file=sys.stderr)

    resultados: list[Medicion] = []
    for modelo in modelos:
        if not modelo.exists():
            resultados.append(Medicion(modelo.name, "llama.cpp", error="fichero no existe"))
            continue
        for etiqueta, binario, ngl in backends:
            print(f"  · {modelo.name}  [{etiqueta}] …", file=sys.stderr, flush=True)
            t0 = time.perf_counter()
            pp, tg, err = _correr_llama_bench(
                binario, modelo, ngl=ngl, n_prompt=n_prompt, n_gen=n_gen, reps=reps)
            dt = time.perf_counter() - t0
            resultados.append(Medicion(modelo.name, etiqueta, pp_ts=pp, tg_ts=tg,
                                       error=err, extra={"wall_s": round(dt, 1)}))
    return resultados


def medir_openvino(modelos: list[Path], *, n_gen: int, prompt: str) -> list[Medicion]:
    """Mide OpenVINO GenAI cargando cada modelo con LlamaVino y contando tokens.

    Solo aplica a arquitecturas que OpenVINO puede ingerir (llama/qwen2/phi3/gemma);
    los demás devuelven el error de carga capturado.
    """
    sys.path.insert(0, str(RAIZ))
    try:
        import openvino_genai as ov_genai  # noqa: F401
        import LlamaVino as lv
    except Exception as exc:  # pragma: no cover - depende del entorno
        return [Medicion(m.name, "OpenVINO (GPU)", error=f"import falló: {exc}")
                for m in modelos]

    resultados: list[Medicion] = []
    for modelo in modelos:
        etiqueta = "OpenVINO (GPU)"
        print(f"  · {modelo.name}  [{etiqueta}] …", file=sys.stderr, flush=True)
        try:
            args = lv.parse_args([
                "-m", str(modelo), "--device", "GPU",
                "--engine", "openvino", "--max-new-tokens", str(n_gen),
            ])
            device = lv.pick_device("GPU")
            t0 = time.perf_counter()
            engine = lv.crear_motor(str(modelo), args, device)
            load_s = time.perf_counter() - t0

            settings = lv.default_gen_settings(args)
            settings["max_new_tokens"] = n_gen
            n_tok = [0]

            def streamer(_sub: str, _cont=n_tok) -> None:
                _cont[0] += 1

            t1 = time.perf_counter()
            lv._motor_generar(engine, [{"role": "user", "content": prompt}],
                              settings, ov_genai, streamer)
            gen_s = time.perf_counter() - t1
            lv.cerrar_motor(engine)
            tg = n_tok[0] / gen_s if gen_s > 0 else None
            resultados.append(Medicion(modelo.name, etiqueta, tg_ts=tg, load_s=load_s,
                                       extra={"tokens": n_tok[0], "gen_s": round(gen_s, 1)}))
        except Exception as exc:  # pragma: no cover
            resultados.append(Medicion(modelo.name, etiqueta, error=str(exc)[:160]))
    return resultados


def registrar_en_perf(mediciones: list[Medicion], modelos: list[Path]) -> list[str]:
    """Escribe en ``perf_db`` la velocidad **de CPU** (la opción por defecto/más rápida).

    Para cada GGUF medido toma su ``tg`` en CPU (tokens/segundo) y reemplaza la fila
    del modelo: registra el motor (``llamacpp``), el formato (``GGUF``), el tiempo de
    preparación del motor en CPU y la velocidad medida. Devuelve líneas de resumen.
    """
    import time

    sys.path.insert(0, str(RAIZ))
    import perf_db  # noqa: E402
    try:
        import LlamaVino as lv  # noqa: E402
    except Exception:  # noqa: BLE001 - si no se puede importar, solo tok/s
        lv = None

    cpu = {m.modelo: m for m in mediciones
           if "CPU" in m.backend and isinstance(m.tg_ts, (int, float))}
    resumen: list[str] = []
    for modelo in modelos:
        med = cpu.get(modelo.name)
        if med is None:
            continue
        load_s = 0.0
        if lv is not None:
            try:
                args = lv.parse_args(["-m", str(modelo), "--cpu"])
                dev = lv.pick_device(args.device)
                t0 = time.perf_counter()
                eng = lv.crear_motor(str(modelo), args, dev, verbose=False)
                load_s = time.perf_counter() - t0
                lv.cerrar_motor(eng)
            except Exception:  # noqa: BLE001 - registro best-effort
                load_s = 0.0
        perf_db.borrar(modelo.name)
        perf_db.registrar_carga(modelo.name, load_s, formato="GGUF", motor="llamacpp")
        perf_db.registrar_generacion(modelo.name, float(med.tg_ts))
        resumen.append(f"  {modelo.name}: {med.tg_ts:.1f} tok/s (CPU) · carga {load_s:.1f}s")
    return resumen


def _fmt(v: float | None) -> str:
    return f"{v:,.1f}" if isinstance(v, (int, float)) else "—"


def render_markdown(mediciones: list[Medicion], meta: dict) -> str:
    """Construye una tabla Markdown comparativa (tok/s; más alto = mejor)."""
    lineas = [
        "# Benchmark de backends — LlamaVINO",
        "",
        f"- Fecha: {meta['fecha']}",
        f"- Carga de trabajo: prompt={meta['n_prompt']} tok · generación={meta['n_gen']} tok"
        f" · repeticiones={meta['reps']}",
        f"- GPU: {meta.get('gpu', 'Intel Iris Xe (Vulkan, uma=1, fp16=1)')}",
        "",
        "**tok/s — más alto = mejor.** `pp` = procesado de prompt (prefill); "
        "`tg` = generación de tokens (lo que se percibe como velocidad).",
        "",
        "| Modelo | Backend | pp (tok/s) | tg (tok/s) | Carga (s) | Nota |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for m in mediciones:
        nota = m.error or (f"{m.extra.get('tokens')} tok" if m.extra.get("tokens") else "")
        lineas.append(
            f"| {m.modelo} | {m.backend} | {_fmt(m.pp_ts)} | {_fmt(m.tg_ts)} "
            f"| {_fmt(m.load_s)} | {nota} |")
    lineas += [
        "",
        "## Cómo añadir la columna SYCL",
        "",
        "El build actual de `vendor/llama.cpp` es **Vulkan** (no incluye `ggml-sycl.dll`).",
        "Para medir SYCL hay que conseguir un `llama-bench.exe` compilado con el backend",
        "SYCL de Intel oneAPI y apuntar la variable de entorno antes de relanzar:",
        "",
        "```powershell",
        "# 1) Instala Intel oneAPI Base Toolkit (incluye DPC++/SYCL y oneMKL).",
        "# 2) Consigue un build SYCL de llama.cpp (compilado con -DGGML_SYCL=ON)",
        "#    o descarga el zip 'sycl' de las releases de llama.cpp.",
        "$env:LLAMA_SYCL_BENCH = 'C:\\\\ruta\\\\al\\\\build-sycl\\\\llama-bench.exe'",
        "python bench/benchmark.py --openvino",
        "```",
        "",
    ]
    return "\n".join(lineas)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark de backends de LlamaVINO.")
    parser.add_argument("modelos", nargs="*", help="GGUF a medir (por defecto, models/gguf/*.gguf).")
    parser.add_argument("-p", "--n-prompt", type=int, default=128, help="Tokens de prompt (pp).")
    parser.add_argument("-n", "--n-gen", type=int, default=64, help="Tokens a generar (tg).")
    parser.add_argument("-r", "--reps", type=int, default=2, help="Repeticiones de llama-bench.")
    parser.add_argument("--openvino", action="store_true",
                        help="Incluye OpenVINO GenAI (solo archs soportadas; más lento de cargar).")
    parser.add_argument("--prompt", default="Explica en tres frases qué es la inferencia de un LLM.",
                        help="Prompt para la medición de OpenVINO.")
    parser.add_argument("--registrar-perf", action="store_true",
                        help="Guarda la velocidad de CPU (la más rápida) en models/llamavino.db.")
    args = parser.parse_args(argv)

    modelos = _descubrir_gguf(args.modelos)
    if not modelos:
        print("No hay GGUF que medir (¿models/gguf vacío?).", file=sys.stderr)
        return 1

    print(f"Modelos: {', '.join(m.name for m in modelos)}", file=sys.stderr)
    print("== llama.cpp (Vulkan / CPU / SYCL) ==", file=sys.stderr)
    mediciones = medir_llamacpp(modelos, n_prompt=args.n_prompt, n_gen=args.n_gen, reps=args.reps)

    if args.openvino:
        print("== OpenVINO GenAI ==", file=sys.stderr)
        mediciones += medir_openvino(modelos, n_gen=args.n_gen, prompt=args.prompt)

    meta = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "n_prompt": args.n_prompt, "n_gen": args.n_gen, "reps": args.reps,
    }
    md = render_markdown(mediciones, meta)
    print("\n" + md)

    if args.registrar_perf:
        print("== Registrando velocidad de CPU en perf_db ==", file=sys.stderr)
        for linea in registrar_en_perf(mediciones, modelos):
            print(linea, file=sys.stderr)

    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    sello = datetime.now().strftime("%Y%m%d-%H%M%S")
    (DIR_RESULTADOS / f"resultados-{sello}.md").write_text(md, encoding="utf-8")
    (DIR_RESULTADOS / f"resultados-{sello}.json").write_text(
        json.dumps([m.__dict__ for m in mediciones], ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n[✓] Guardado en bench/resultados-{sello}.md / .json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
