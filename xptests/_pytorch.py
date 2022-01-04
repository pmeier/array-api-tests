import functools
import math

import torch

from torch import arange, finfo, full, linspace

sign = functools.partial(math.copysign, 1)

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

array_module.asarray = torch.as_tensor
array_module.equal = torch.eq


class SizeWrapper(int):
    def __new__(cls, tensor):
        obj = int.__new__(cls, tensor.numel())
        obj._shape = tensor.shape
        return obj

    def __call__(self):
        return self._shape


torch.Tensor.size = property(lambda self: SizeWrapper(self))


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
            raise TypeError("`torch.eye` currently does not support the 'k' parameter")
        if m is None:
            return eye(n, **kwargs)
        else:
            return eye(n, m, **kwargs)

    return wrapper


array_module.eye = patch_eye(torch.eye)


def patched_arange(arg1, arg2=None, step=1, **kwargs):
    start, end = (0, arg1) if arg2 is None else (arg1, arg2)
    if sign(end - start) != sign(step):
        start, end, step = promote_scalars(0, 0, 1, to=promote_type(start, end, step))

    return arange(start, end, step, **kwargs)


array_module.arange = patched_arange


def patched_linspace(*args, endpoint=True, **kwargs):
    if not endpoint:
        raise TypeError(
            "`torch.linspace` currently does not support the 'endpoint' parameter"
        )
    return linspace(*args, **kwargs)


array_module.linspace = patched_linspace
