"""Tiny YAML subset for the bundled configs. Used when PyYAML is not installed."""

from __future__ import annotations

from typing import Any


def safe_load(text: str) -> Any:
    lines = [_strip_comment(line.rstrip("\n")) for line in text.splitlines()]
    body = [(i + 1, line) for i, line in enumerate(lines) if line.strip()]
    value, _index = _parse_block(body, 0, 0)
    return value


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index].rstrip()
    return line.rstrip()


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _parse_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index
    _no, line = lines[index]
    if _indent(line) != indent:
        return {}, index
    if line.lstrip().startswith("- "):
        return _parse_list(lines, index, indent)
    return _parse_map(lines, index, indent)


def _parse_map(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[dict[str, Any], int]:
    out: dict[str, Any] = {}
    while index < len(lines):
        _no, line = lines[index]
        cur = _indent(line)
        if cur < indent:
            break
        if cur > indent:
            raise ValueError(f"bad indent on line {_no}")
        raw = line.strip()
        if raw.startswith("- "):
            break
        if ":" not in raw:
            raise ValueError(f"expected key: on line {_no}")
        key, rest = raw.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        index += 1
        if rest:
            out[key] = _parse_flow(rest)
            continue
        if index >= len(lines) or _indent(lines[index][1]) <= indent:
            out[key] = {}
            continue
        child, index = _parse_block(lines, index, _indent(lines[index][1]))
        out[key] = child
    return out, index


def _parse_list(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[list[Any], int]:
    out: list[Any] = []
    while index < len(lines):
        _no, line = lines[index]
        cur = _indent(line)
        if cur < indent:
            break
        if cur > indent:
            raise ValueError(f"bad indent on line {_no}")
        raw = line.strip()
        if not raw.startswith("- "):
            break
        rest = raw[2:].strip()
        index += 1
        if rest:
            if ":" in rest and not rest.startswith(("[", "{", "'", '"')):
                key, value = rest.split(":", 1)
                item: dict[str, Any] = {key.strip(): _parse_flow(value.strip())}
                if index < len(lines) and _indent(lines[index][1]) > indent:
                    child, index = _parse_block(lines, index, _indent(lines[index][1]))
                    if isinstance(child, dict):
                        item.update(child)
                out.append(item)
                continue
            out.append(_parse_flow(rest))
            continue
        if index >= len(lines) or _indent(lines[index][1]) <= indent:
            out.append({})
            continue
        child, index = _parse_block(lines, index, _indent(lines[index][1]))
        out.append(child)
    return out, index


def _parse_flow(text: str) -> Any:
    if text.startswith("[") and text.endswith("]"):
        return [_parse_flow(part.strip()) for part in _split_flow(text[1:-1]) if part.strip() != ""]
    if text.startswith("{") and text.endswith("}"):
        mapping: dict[str, Any] = {}
        for part in _split_flow(text[1:-1]):
            if not part.strip():
                continue
            key, rest = part.split(":", 1)
            mapping[key.strip()] = _parse_flow(rest.strip())
        return mapping
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    lowered = text.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"null", "~", ""}:
        return None
    try:
        if any(char in text for char in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _split_flow(text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    quote = ""
    for char in text:
        if quote:
            buf.append(char)
            if char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
            buf.append(char)
            continue
        if char in {"[", "{"}:
            depth += 1
            buf.append(char)
            continue
        if char in {"]", "}"}:
            depth -= 1
            buf.append(char)
            continue
        if char == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(char)
    if buf:
        parts.append("".join(buf))
    return parts
