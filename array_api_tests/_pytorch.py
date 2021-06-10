import math

import torch

__all__ = ["array_module"]

array_module = torch

array_module.asarray = torch.as_tensor
array_module.equal = torch.eq

for constant in ("e", "inf", "nan", "pi"):
    setattr(array_module, constant, getattr(math, constant))

array_module.Tensor.size = property(lambda self: self.numel())


def patch_full(full):
    def wrapper(shape, fill_value, dtype=None, **kwargs):
        finfo = torch.finfo(dtype or torch.float32)

        if fill_value > finfo.max:
            fill_value = math.inf
        elif fill_value < finfo.min:
            fill_value = -math.inf

        return full(shape, fill_value, dtype=dtype, **kwargs)

    return wrapper


array_module.full = patch_full(array_module.full)
