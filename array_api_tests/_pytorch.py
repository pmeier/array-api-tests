import torch

__all__ = ["array_module"]

array_module = torch

array_module.asarray = torch.as_tensor
array_module.equal = torch.eq

array_module.e = torch.tensor(2.718281828459045)
array_module.inf = torch.tensor(float("Inf"))
array_module.nan = torch.tensor(float("NaN"))
array_module.pi = torch.tensor(3.141592653589793)

array_module.Tensor.size = property(lambda self: self.numel())
