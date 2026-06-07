#!/usr/bin/env python
"""Base de datos SQLite con el rendimiento de cada modelo en esta máquina.

Guarda, por modelo, el tiempo de carga (segundos) y la velocidad de generación
(tokens/segundo, media móvil), además del formato/motor y la última vez usado.
Se actualiza cada vez que se carga o se genera con un modelo, para poder elegir
el «más potente entre los más rápidos» con datos reales. Sin dependencias.
"""

from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path

DB_PATH = Path("models") / "llamavino.db"

_ESQUEMA = """
CREATE TABLE IF NOT EXISTS modelos (
    modelo       TEXT PRIMARY KEY,   -- nombre del fichero/directorio del modelo
    formato      TEXT,               -- GGUF / IR
    motor        TEXT,               -- openvino / llamacpp / llamaserver
    load_seconds REAL,               -- último tiempo de carga
    tok_s        REAL,               -- velocidad media (tokens/segundo)
    runs         INTEGER DEFAULT 0,  -- nº de generaciones registradas
    last_used    TEXT                -- ISO-8601 de la última vez usado
)
"""


def _ahora() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _conn(db_path=None) -> sqlite3.Connection:
    ruta = Path(db_path) if db_path else DB_PATH
    ruta.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(ruta))
    con.row_factory = sqlite3.Row
    con.execute(_ESQUEMA)
    return con


def registrar_carga(modelo: str, segundos: float, *, formato: str | None = None,
                    motor: str | None = None, db_path=None) -> None:
    """Registra/actualiza el tiempo de carga (y formato/motor) de ``modelo``."""
    con = _conn(db_path)
    try:
        with con:
            con.execute(
                """INSERT INTO modelos (modelo, formato, motor, load_seconds, last_used)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(modelo) DO UPDATE SET
                       load_seconds = excluded.load_seconds,
                       formato = COALESCE(excluded.formato, modelos.formato),
                       motor   = COALESCE(excluded.motor, modelos.motor),
                       last_used = excluded.last_used""",
                (modelo, formato, motor, float(segundos), _ahora()))
    finally:
        con.close()


def registrar_generacion(modelo: str, tok_s: float, *, db_path=None) -> None:
    """Actualiza la velocidad media (tok/s) de ``modelo`` con una nueva medición."""
    con = _conn(db_path)
    try:
        with con:
            fila = con.execute(
                "SELECT tok_s, runs FROM modelos WHERE modelo = ?", (modelo,)
            ).fetchone()
            if fila and fila["tok_s"] is not None and fila["runs"]:
                media = (fila["tok_s"] * fila["runs"] + tok_s) / (fila["runs"] + 1)
                con.execute(
                    "UPDATE modelos SET tok_s = ?, runs = runs + 1, last_used = ? "
                    "WHERE modelo = ?", (media, _ahora(), modelo))
            else:
                con.execute(
                    """INSERT INTO modelos (modelo, tok_s, runs, last_used)
                       VALUES (?, ?, 1, ?)
                       ON CONFLICT(modelo) DO UPDATE SET
                           tok_s = excluded.tok_s, runs = 1, last_used = excluded.last_used""",
                    (modelo, float(tok_s), _ahora()))
    finally:
        con.close()


def obtener(modelo: str, *, db_path=None) -> dict | None:
    """Devuelve el registro de rendimiento de ``modelo`` (o None)."""
    con = _conn(db_path)
    try:
        fila = con.execute("SELECT * FROM modelos WHERE modelo = ?", (modelo,)).fetchone()
        return dict(fila) if fila else None
    finally:
        con.close()


def todos(*, db_path=None) -> dict[str, dict]:
    """Devuelve ``{modelo: registro}`` con todo el rendimiento almacenado."""
    con = _conn(db_path)
    try:
        return {f["modelo"]: dict(f) for f in con.execute("SELECT * FROM modelos")}
    finally:
        con.close()


def borrar(modelo: str, *, db_path=None) -> None:
    """Elimina el registro de rendimiento de ``modelo`` (al borrarlo del disco)."""
    con = _conn(db_path)
    try:
        with con:
            con.execute("DELETE FROM modelos WHERE modelo = ?", (modelo,))
    finally:
        con.close()
