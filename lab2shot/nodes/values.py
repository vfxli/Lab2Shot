"""Node ids that a wire drawn out of a value parameter offers first: what the node menu recommends, not the values
themselves (their type, unit and per-frame shape are lab2shot/data/values.py)."""

from __future__ import annotations

from ..data.values import BOOL, FLOAT, INT, TEXT, VECTOR

# the constant node of each type: what the node menu offers first for a wire drawn out of a value parameter
CONSTANT_NODES = {FLOAT: "value_float", INT: "value_int", BOOL: "value_toggle", VECTOR: "value_vector",
                  TEXT: "value_string"}
