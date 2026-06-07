"""Tests de code_structure (Tree-sitter): símbolos, extracción y esquema."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import code_structure as cs  # noqa: E402

SOURCE = '''\
import os


def saludar(nombre):
    return f"hola {nombre}"


class Animal:
    """Un animal."""

    def __init__(self, patas):
        self.patas = patas

    @property
    def descripcion(self):
        return f"{self.patas} patas"


def fabrica():
    def interna():
        return 1
    return interna
'''


class ListSymbolsTests(unittest.TestCase):
    def test_finds_all_symbols(self):
        names = {s.qualified_name for s in cs.list_symbols(SOURCE)}
        self.assertEqual(
            names,
            {
                "saludar",
                "Animal",
                "Animal.__init__",
                "Animal.descripcion",
                "fabrica",
                "fabrica.interna",
            },
        )

    def test_kinds(self):
        by_name = {s.qualified_name: s.kind for s in cs.list_symbols(SOURCE)}
        self.assertEqual(by_name["Animal"], "class")
        self.assertEqual(by_name["saludar"], "function")
        self.assertEqual(by_name["Animal.__init__"], "function")

    def test_line_spans(self):
        sym = {s.qualified_name: s for s in cs.list_symbols(SOURCE)}["saludar"]
        self.assertEqual(sym.start_line, 4)
        self.assertEqual(sym.end_line, 5)


class ExtractSymbolTests(unittest.TestCase):
    def test_extract_function(self):
        frag = cs.extract_symbol(SOURCE, "saludar")
        self.assertEqual(frag, 'def saludar(nombre):\n    return f"hola {nombre}"')

    def test_extract_nested_method(self):
        frag = cs.extract_symbol(SOURCE, "Animal.__init__")
        self.assertEqual(frag, "def __init__(self, patas):\n        self.patas = patas")

    def test_extract_includes_decorator(self):
        frag = cs.extract_symbol(SOURCE, "Animal.descripcion")
        self.assertTrue(frag.startswith("@property"))
        self.assertIn("def descripcion(self):", frag)

    def test_extract_whole_class(self):
        frag = cs.extract_symbol(SOURCE, "Animal")
        self.assertTrue(frag.startswith("class Animal:"))
        self.assertIn("def descripcion", frag)

    def test_extract_inner_function(self):
        frag = cs.extract_symbol(SOURCE, "fabrica.interna")
        self.assertEqual(frag, "def interna():\n        return 1")

    def test_not_found_raises(self):
        with self.assertRaises(cs.StructureError):
            cs.extract_symbol(SOURCE, "noExiste")


class OutlineTests(unittest.TestCase):
    def test_outline_has_indentation_and_names(self):
        text = cs.outline(SOURCE)
        self.assertIn("def saludar", text)
        self.assertIn("class Animal", text)
        # El método va indentado bajo la clase.
        self.assertIn("  def descripcion", text)


class LanguageDetectionTests(unittest.TestCase):
    def test_language_for_path(self):
        self.assertEqual(cs.language_for_path("x/y.py"), "python")

    def test_unknown_extension_raises(self):
        with self.assertRaises(cs.StructureError):
            cs.language_for_path("x/y.rs")


if __name__ == "__main__":
    unittest.main(verbosity=2)
