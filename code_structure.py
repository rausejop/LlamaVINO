#!/usr/bin/env python
"""Análisis estructural de código con Tree-sitter para LlamaVino.

En lugar de tratar el código como texto plano, este módulo construye un árbol
sintáctico concreto (CST) con [Tree-sitter](https://tree-sitter.github.io/) y
permite extraer **funciones, clases o scopes exactos** por su nombre. Así se le
puede enviar al modelo solo la porción precisa que necesita ver, en vez del
fichero entero.

Hoy soporta Python (grammar `tree-sitter-python`); la estructura está preparada
para añadir más lenguajes registrando su grammar en ``_LANGUAGES``.

Las funciones son puras sobre texto: reciben el código fuente y devuelven símbolos
o fragmentos, sin tocar el disco.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_DEFINITION_KINDS = {
    "function_definition": "function",
    "class_definition": "class",
}


class StructureError(Exception):
    """Error al analizar el código o al localizar un símbolo."""


@dataclass
class Symbol:
    """Un símbolo definido en el código (función o clase).

    Attributes:
      kind: ``"function"`` o ``"class"``.
      name: Nombre simple del símbolo.
      qualified_name: Nombre con su ruta de scopes (p. ej. ``Clase.metodo``).
      start_line: Primera línea (1-based), incluyendo decoradores.
      end_line: Última línea (1-based).
      start_byte: Offset de byte inicial en el fuente.
      end_byte: Offset de byte final.
    """

    kind: str
    name: str
    qualified_name: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int


# Caché de objetos Language por nombre de lenguaje.
_LANGUAGES: dict[str, object] = {}


def _get_language(language: str):
    """Devuelve (y cachea) el objeto Language de Tree-sitter para un lenguaje.

    Raises:
      StructureError: Si el lenguaje no está soportado o falta su grammar.
    """
    if language in _LANGUAGES:
        return _LANGUAGES[language]
    try:
        from tree_sitter import Language
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise StructureError(
            "tree-sitter no está instalado: "
            "python -m pip install tree-sitter tree-sitter-python"
        ) from exc

    if language == "python":
        try:
            import tree_sitter_python as ts_python
        except ImportError as exc:  # pragma: no cover
            raise StructureError("falta tree-sitter-python") from exc
        lang = Language(ts_python.language())
    else:
        raise StructureError(f"lenguaje no soportado: {language!r}")

    _LANGUAGES[language] = lang
    return lang


def language_for_path(path: str | Path) -> str:
    """Infiere el lenguaje a partir de la extensión del fichero.

    Raises:
      StructureError: Si la extensión no tiene un lenguaje soportado.
    """
    suffix = Path(path).suffix.lower()
    mapping = {".py": "python", ".pyi": "python"}
    if suffix not in mapping:
        raise StructureError(f"extensión sin lenguaje soportado: {suffix!r}")
    return mapping[suffix]


def _parse(source: str, language: str):
    """Parsea el fuente y devuelve ``(root_node, source_bytes)``."""
    from tree_sitter import Parser

    parser = Parser(_get_language(language))
    data = source.encode("utf-8")
    return parser.parse(data).root_node, data


def _span_node(node):
    """Nodo cuyo span usar: incluye decoradores si los hay."""
    parent = node.parent
    if parent is not None and parent.type == "decorated_definition":
        return parent
    return node


def list_symbols(source: str, language: str = "python") -> list[Symbol]:
    """Lista las funciones y clases del fuente, incluidas las anidadas.

    Args:
      source: Código fuente.
      language: Lenguaje del grammar (por defecto ``"python"``).

    Returns:
      Lista de :class:`Symbol` en orden de aparición. ``qualified_name`` refleja
      el anidamiento (``Clase.metodo``, ``funcion.interna``).
    """
    root, _ = _parse(source, language)
    symbols: list[Symbol] = []

    def walk(node, scope: list[str]) -> None:
        for child in node.children:
            kind = _DEFINITION_KINDS.get(child.type)
            if kind is not None:
                name_node = child.child_by_field_name("name")
                name = name_node.text.decode("utf-8") if name_node else "<anon>"
                qualified = ".".join([*scope, name])
                span = _span_node(child)
                symbols.append(
                    Symbol(
                        kind=kind,
                        name=name,
                        qualified_name=qualified,
                        start_line=span.start_point[0] + 1,
                        end_line=span.end_point[0] + 1,
                        start_byte=span.start_byte,
                        end_byte=span.end_byte,
                    )
                )
                walk(child, [*scope, name])
            else:
                walk(child, scope)

    walk(root, [])
    return symbols


def extract_symbol(source: str, qualified_name: str, language: str = "python") -> str:
    """Devuelve el texto fuente exacto de un símbolo por su nombre cualificado.

    Args:
      source: Código fuente.
      qualified_name: Nombre con scopes, p. ej. ``"Clase.metodo"``.
      language: Lenguaje del grammar.

    Returns:
      El fragmento de código del símbolo (incluyendo decoradores).

    Raises:
      StructureError: Si no existe un símbolo con ese nombre.
    """
    data = source.encode("utf-8")
    for symbol in list_symbols(source, language):
        if symbol.qualified_name == qualified_name:
            return data[symbol.start_byte:symbol.end_byte].decode("utf-8")
    raise StructureError(f"símbolo no encontrado: {qualified_name!r}")


def outline(source: str, language: str = "python") -> str:
    """Devuelve un esquema indentado de los símbolos del fuente.

    Útil para darle al modelo el mapa del fichero sin todo el cuerpo.
    """
    lines: list[str] = []
    for symbol in list_symbols(source, language):
        depth = symbol.qualified_name.count(".")
        prefix = "  " * depth
        marker = "def" if symbol.kind == "function" else "class"
        lines.append(
            f"{prefix}{marker} {symbol.name}  "
            f"[L{symbol.start_line}-{symbol.end_line}]"
        )
    return "\n".join(lines)
