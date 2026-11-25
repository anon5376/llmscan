"""Versioned formula-safety encoding for llmscan CSV files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

ENCODING_COLUMN = "llmscan_csv_encoding"
ENCODING_MARKER = "apostrophe-v1"
_ESCAPE_PREFIXES = "=+-@\t\r\n'"


def csv_fieldnames(fields: Sequence[str]) -> list[str]:
    return [*fields, ENCODING_COLUMN]


def encode_row(row: Mapping[str, object]) -> dict[str, object]:
    encoded = {
        key: "'" + value if isinstance(value, str) and value and value[0] in _ESCAPE_PREFIXES else value
        for key, value in row.items()
    }
    encoded[ENCODING_COLUMN] = ENCODING_MARKER
    return encoded


def decode_row(row: Mapping[str, object]) -> dict[str, object]:
    if row.get(ENCODING_COLUMN) != ENCODING_MARKER:
        return dict(row)
    decoded = {}
    for key, value in row.items():
        if key == ENCODING_COLUMN:
            continue
        if isinstance(value, str) and len(value) > 1 and value[0] == "'" and value[1] in _ESCAPE_PREFIXES:
            value = value[1:]
        decoded[key] = value
    return decoded
