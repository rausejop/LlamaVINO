"""Pruebas de la BD de rendimiento SQLite (perf_db)."""

import tempfile
import unittest
from pathlib import Path

import perf_db


class PerfDBTests(unittest.TestCase):
    def _db(self) -> Path:
        base = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(base, ignore_errors=True))
        return base / "perf.db"

    def test_carga_y_generacion(self):
        db = self._db()
        perf_db.registrar_carga("m.gguf", 12.5, formato="GGUF", motor="openvino",
                                db_path=db)
        perf_db.registrar_generacion("m.gguf", 10.0, db_path=db)
        perf_db.registrar_generacion("m.gguf", 20.0, db_path=db)
        r = perf_db.obtener("m.gguf", db_path=db)
        self.assertEqual(r["load_seconds"], 12.5)
        self.assertEqual(r["tok_s"], 15.0)   # media de 10 y 20
        self.assertEqual(r["runs"], 2)
        self.assertEqual(r["formato"], "GGUF")

    def test_generacion_sin_carga_previa(self):
        db = self._db()
        perf_db.registrar_generacion("x.gguf", 8.0, db_path=db)
        self.assertEqual(perf_db.obtener("x.gguf", db_path=db)["tok_s"], 8.0)

    def test_todos_y_borrar(self):
        db = self._db()
        perf_db.registrar_carga("a", 1.0, db_path=db)
        perf_db.registrar_carga("b", 2.0, db_path=db)
        self.assertEqual(set(perf_db.todos(db_path=db)), {"a", "b"})
        perf_db.borrar("a", db_path=db)
        self.assertNotIn("a", perf_db.todos(db_path=db))


if __name__ == "__main__":
    unittest.main()
