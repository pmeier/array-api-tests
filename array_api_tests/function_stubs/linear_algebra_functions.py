"""
Function stubs for linear algebra functions.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/linear_algebra_functions.md
"""

from __future__ import annotations

from ._types import Optional, Tuple, Union, array
from collections.abc import Sequence

def einsum():
    pass

def matmul(x1: array, x2: array, /) -> array:
    pass

def tensordot(x1: array, x2: array, /, *, axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2) -> array:
    pass

def transpose(x: array, /, *, axes: Optional[Tuple[int, ...]] = None) -> array:
    pass

def vecdot(x1: array, x2: array, /, *, axis: Optional[int] = None) -> array:
    pass

__all__ = ['einsum', 'matmul', 'tensordot', 'transpose', 'vecdot']
