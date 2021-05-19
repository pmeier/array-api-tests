import torch

__all__ = ["array_module"]

array_module = torch
array_module.asarray = torch.as_tensor
array_module.inf = torch.tensor(float("Inf"))
array_module.equal = torch.eq
