"""Tests de edit_formats: unified diff, bloques Aider y escritura segura."""

from __future__ import annotations

import difflib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import edit_formats as ef  # noqa: E402


def make_unified(before: str, after: str) -> str:
    """Genera un unified diff entre dos textos con difflib."""
    diff = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile="a/f",
        tofile="b/f",
        lineterm="",
    )
    return "\n".join(diff) + "\n"


class UnifiedDiffTests(unittest.TestCase):
    def test_roundtrip_simple(self):
        before = "uno\ndos\ntres\n"
        after = "uno\nDOS\ntres\n"
        diff = make_unified(before, after)
        self.assertEqual(ef.apply_unified_diff(diff, before), after)

    def test_roundtrip_multi_hunk(self):
        before = "\n".join(f"linea {i}" for i in range(1, 31)) + "\n"
        after = before.replace("linea 3", "LINEA TRES").replace("linea 27", "LINEA 27!")
        diff = make_unified(before, after)
        self.assertEqual(ef.apply_unified_diff(diff, before), after)

    def test_roundtrip_add_and_remove(self):
        before = "a\nb\nc\nd\ne\n"
        after = "a\nc\nNUEVA\nd\ne\nf\n"
        diff = make_unified(before, after)
        self.assertEqual(ef.apply_unified_diff(diff, before), after)

    def test_preserves_no_trailing_newline(self):
        before = "a\nb\nc"  # sin salto final
        after = "a\nB\nc"
        diff = make_unified(before, after)
        result = ef.apply_unified_diff(diff, before)
        self.assertEqual(result, after)
        self.assertFalse(result.endswith("\n"))

    def test_fuzzy_offset(self):
        # El diff se generó sobre un fichero más corto; al original real le
        # sobran líneas al principio. El localizador con fuzz debe encontrarlo.
        base = "x\ny\nz\n"
        after = "x\nY\nz\n"
        diff = make_unified(base, after)
        shifted = "cabecera\n" + base
        expected = "cabecera\n" + after
        self.assertEqual(ef.apply_unified_diff(diff, shifted), expected)

    def test_context_mismatch_raises(self):
        diff = make_unified("a\nb\nc\n", "a\nB\nc\n")
        with self.assertRaises(ef.PatchError):
            ef.apply_unified_diff(diff, "completamente\ndistinto\n")

    def test_empty_diff_raises(self):
        with self.assertRaises(ef.PatchError):
            ef.apply_unified_diff("", "a\n")


class AiderBlockTests(unittest.TestCase):
    def test_basic_replace(self):
        content = "def f():\n    return 1\n"
        block = (
            "app.py\n"
            "<<<<<<< SEARCH\n"
            "    return 1\n"
            "=======\n"
            "    return 2\n"
            ">>>>>>> REPLACE\n"
        )
        (parsed,) = ef.parse_search_replace(block)
        self.assertEqual(parsed.path, "app.py")
        self.assertEqual(ef.apply_search_replace(content, parsed), "def f():\n    return 2\n")

    def test_new_file_empty_search(self):
        block = "nuevo.py\n<<<<<<< SEARCH\n=======\nhola = 1\n>>>>>>> REPLACE\n"
        (parsed,) = ef.parse_search_replace(block)
        self.assertEqual(ef.apply_search_replace("", parsed), "hola = 1")

    def test_not_found_raises(self):
        block = "<<<<<<< SEARCH\nno existe\n=======\nx\n>>>>>>> REPLACE\n"
        (parsed,) = ef.parse_search_replace(block)
        with self.assertRaises(ef.PatchError):
            ef.apply_search_replace("otra cosa\n", parsed)

    def test_flexible_trailing_whitespace(self):
        content = "linea_a   \nlinea_b\n"  # espacios sobrantes en el original
        block = "<<<<<<< SEARCH\nlinea_a\nlinea_b\n=======\nX\nY\n>>>>>>> REPLACE\n"
        (parsed,) = ef.parse_search_replace(block)
        # El contenido tenía salto final; el reemplazo lo preserva.
        self.assertEqual(ef.apply_search_replace(content, parsed), "X\nY\n")

    def test_multiple_blocks(self):
        text = (
            "<<<<<<< SEARCH\nuno\n=======\nUNO\n>>>>>>> REPLACE\n"
            "<<<<<<< SEARCH\ndos\n=======\nDOS\n>>>>>>> REPLACE\n"
        )
        blocks = ef.parse_search_replace(text)
        self.assertEqual(len(blocks), 2)
        out = "uno\ndos\ntres"
        for blk in blocks:
            out = ef.apply_search_replace(out, blk)
        self.assertEqual(out, "UNO\nDOS\ntres")

    def test_unclosed_block_raises(self):
        with self.assertRaises(ef.PatchError):
            ef.parse_search_replace("<<<<<<< SEARCH\nx\n")


class FileIOTests(unittest.TestCase):
    def test_apply_unified_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.txt"
            path.write_text("a\nb\nc\n", encoding="utf-8")
            diff = make_unified("a\nb\nc\n", "a\nB\nc\n")
            ef.apply_patch_to_file(path, diff, fmt="unified")
            self.assertEqual(path.read_text(encoding="utf-8"), "a\nB\nc\n")

    def test_apply_aider_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.txt"
            path.write_text("valor = 1\n", encoding="utf-8")
            block = "<<<<<<< SEARCH\nvalor = 1\n=======\nvalor = 99\n>>>>>>> REPLACE\n"
            ef.apply_patch_to_file(path, block, fmt="aider")
            self.assertEqual(path.read_text(encoding="utf-8"), "valor = 99\n")

    def test_write_with_retry_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "out.txt"
            path.parent.mkdir(parents=True)
            ef.write_text_with_retry(path, "ñandú\n")
            self.assertEqual(ef.read_text(path), "ñandú\n")

    def test_unknown_format_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.txt"
            path.write_text("x\n", encoding="utf-8")
            with self.assertRaises(ef.PatchError):
                ef.apply_patch_to_file(path, "x", fmt="xyz")


if __name__ == "__main__":
    unittest.main(verbosity=2)
