"""
Function stubs for linear algebra functions (Extension).

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/linear_algebra_functions.md
"""

from __future__ import annotations

from ._types import Literal, Optional, Tuple, Union, array
from .constants import inf
from collections.abc import Sequence

def cholesky(x: array, /, *, upper: bool = False) -> array:
    pass

def cross(x1: array, x2: array, /, *, axis: int = -1) -> array:
    pass

def det(x: array, /) -> array:
    pass

def diagonal(x: array, /, *, axis1: int = 0, axis2: int = 1, offset: int = 0) -> array:
    pass

def eig():
    pass

def eigh(x: array, /, *, upper: bool = False) -> Tuple[array]:
    pass

def eigvals():
    pass

def eigvalsh(x: array, /, *, upper: bool = False) -> array:
    pass

def einsum():
    pass

def inv(x: array, /) -> array:
    pass

def lstsq(x1: array, x2: array, /, *, rtol: Optional[Union[float, array]] = None) -> Tuple[array, array, array, array]:
    pass

def matmul(x1: array, x2: array, /) -> array:
    pass

def matrix_power(x: array, n: int, /) -> array:
    pass

def matrix_rank(x: array, /, *, rtol: Optional[Union[float, array]] = None) -> array:
    pass

def norm(x: array, /, *, axis: Optional[Union[int, Tuple[int, int]]] = None, keepdims: bool = False, ord: Optional[Union[int, float, Literal[inf, -inf, 'fro', 'nuc']]] = None) -> array:
    pass

def outer(x1: array, x2: array, /) -> array:
    pass

def pinv(x: array, /, *, rtol: Optional[Union[float, array]] = None) -> array:
    pass

def qr(x: array, /, *, mode: str = 'reduced') -> Tuple[array, array]:
    pass

def slogdet(x: array, /) -> Tuple[array, array]:
    pass

def solve(x1: array, x2: array, /) -> array:
    pass

def svd(x: array, /, *, full_matrices: bool = True) -> Union[array, Tuple[array, ...]]:
    pass

def tensordot(x1: array, x2: array, /, *, axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2) -> array:
    pass

def svdvals(x: array, /) -> Union[array, Tuple[array, ...]]:
    pass

def trace(x: array, /, *, axis1: int = 0, axis2: int = 1, offset: int = 0) -> array:
    pass

def transpose(x: array, /, *, axes: Optional[Tuple[int, ...]] = None) -> array:
    pass

def vecdot(x1: array, x2: array, /, *, axis: Optional[int] = None) -> array:
    pass

__all__ = ['cholesky', 'cross', 'det', 'diagonal', 'eig', 'eigh', 'eigvals', 'eigvalsh', 'einsum', 'inv', 'lstsq', 'matmul', 'matrix_power', 'matrix_rank', 'norm', 'outer', 'pinv', 'qr', 'slogdet', 'solve', 'svd', 'tensordot', 'svdvals', 'trace', 'transpose', 'vecdot']
