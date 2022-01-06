import functools
import math

from collections import namedtuple

import torch

from torch import arange, asarray, finfo, full, linspace, sort, unique

sign = functools.partial(math.copysign, 1)


class ArrayAPIDivergence(Exception):
    pass


TYPE_RANKS = {
    bool: 0,
    int: 1,
    float: 2,
    complex: 3,
}


def promote_type(*scalars):
    return max([type(scalar) for scalar in scalars], key=TYPE_RANKS.get)


def promote_scalars(*scalars, to=None):
    if not to:
        to = promote_type(scalars)
    return tuple(to(scalar) for scalar in scalars)


array_module = torch

array_module.astype = torch.Tensor.to


class SizeWrapper(int):
    def __new__(cls, tensor):
        obj = int.__new__(cls, tensor.numel())
        obj._shape = tensor.shape
        return obj

    def __call__(self, dim=None):
        if dim is None:
            return self._shape

        return self._shape[dim]


array_module.Tensor.size = property(lambda self: SizeWrapper(self))


patched_equal = torch.eq
array_module.equal = patched_equal


def patched_asarray(obj, *, dtype=None, **kwargs):
    array = asarray(obj, dtype=dtype, **kwargs)
    if dtype is None and type(obj) in TYPE_RANKS:
        array = array.to(torch.tensor(type(obj)()).dtype)
    return array


array_module.asarray = patched_asarray


def patched_full(shape, *args, **kwargs):
    return full((shape,) if isinstance(shape, int) else shape, *args, **kwargs)


array_module.full = patched_full


class PatchedFinfo:
    def __init__(self, *args, **kwargs):
        self._info = finfo(*args, **kwargs)

    def __getattr__(self, item):
        if item == "smallest_normal":
            return self._info.tiny
        else:
            return getattr(self._info, item)


array_module.finfo = PatchedFinfo


def patch_eye(eye):
    def wrapper(n, m=None, *, k=0, **kwargs):
        if k != 0:
            raise ArrayAPIDivergence(
                "`torch.eye` currently does not support the 'k' parameter"
            )
        if m is None:
            return eye(n, **kwargs)
        else:
            return eye(n, m, **kwargs)

    return wrapper


array_module.eye = patch_eye(torch.eye)


def patched_arange(arg1, arg2=None, step=1, **kwargs):
    start, end = (0, arg1) if arg2 is None else (arg1, arg2)
    return arange(start, end, step, **kwargs)


array_module.arange = patched_arange


def patched_linspace(*args, endpoint=True, **kwargs):
    if not endpoint:
        raise ArrayAPIDivergence(
            "`torch.linspace` currently does not support the 'endpoint' parameter"
        )
    return linspace(*args, **kwargs)


array_module.linspace = patched_linspace


def patched_unique_counts(*args, **kwargs):
    return namedtuple("UniqueCounts", ("values", "counts"))._make(
        unique(*args, return_counts=True, **kwargs)
    )


def patched_unique_inverse(*args, **kwargs):
    return namedtuple("UniqueInverse", ("values", "inverse_indices"))._make(
        unique(*args, return_inverse=True, **kwargs)
    )


array_module.unique_counts = patched_unique_counts
array_module.unique_inverse = patched_unique_inverse
array_module.unique_values = unique


def patched_sort(*args, **kwargs):
    return sort(*args, **kwargs).values


def patched_argsort(*args, **kwargs):
    return sort(*args, **kwargs).indices


array_module.sort = patched_sort
array_module.argsort = patched_argsort
