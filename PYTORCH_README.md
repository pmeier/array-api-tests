# Array API Standard Test Suite for PyTorch

This documents all deviations of [PyTorch](https://pytorch.org) from the array API specification identified by the test suite.

Array API specification commit: [85b5499](https://github.com/data-apis/array-api/commit/85b549947e41d3a4b65fa3b17f1a95515a35e11c)

PyTorch version: `1.11.0.dev20220103`

## Unsigned integer data types

Reference: [#58734](https://github.com/pytorch/pytorch/issues/58734)

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/data_types.html) four unsigned integer data types:

1. `uint8`,
2. `uint16`,
3. `uint32`, and
4. `uint64`.

Of those, PyTorch [only supports `uint8`](https://pytorch.org/docs/stable/tensor_attributes.html#torch-dtype).

## Inter-category type promotion involving 0d-array's

Reference: [#58736](https://github.com/pytorch/pytorch/issues/58736)

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/type_promotion.html) clear type promotion rules, that are independent of the array size and values:

![Type promotion stipulated by the array API specifictation](https://data-apis.org/array-api/latest/_images/dtype_promotion_lattice.png)

PyTorch mostly adheres to this with one exception: Within a dtype category (integral, floating, complex) 0d-tensors do not participate in type promotion:

```python
import torch

dtype_categories = (
    (torch.int8, torch.uint8, torch.int32, torch.int64),
    (torch.float16, torch.bfloat16, torch.float32, torch.float64),
    (torch.complex32, torch.complex64, torch.complex128),
)

for dtypes in dtype_categories:
    for idx in range(len(dtypes) - 1):
        dtype_nd = dtypes[idx]
        for dtype_0d in dtypes[idx + 1:]:
            a = torch.empty((1,), dtype=dtype_nd)
            b = torch.empty((), dtype=dtype_0d)
            print(f"{a.dtype} + {b.dtype} = {torch.result_type(a, b)}")
```

```
torch.int8 + torch.uint8 = torch.int8
torch.int8 + torch.int32 = torch.int8
torch.int8 + torch.int64 = torch.int8
torch.uint8 + torch.int32 = torch.uint8
torch.uint8 + torch.int64 = torch.uint8
torch.int32 + torch.int64 = torch.int32
torch.float16 + torch.bfloat16 = torch.float16
torch.float16 + torch.float32 = torch.float16
torch.float16 + torch.float64 = torch.float16
torch.bfloat16 + torch.float32 = torch.bfloat16
torch.bfloat16 + torch.float64 = torch.bfloat16
torch.float32 + torch.float64 = torch.float32
torch.complex32 + torch.complex64 = torch.complex32
torch.complex32 + torch.complex128 = torch.complex32
torch.complex64 + torch.complex128 = torch.complex64
```

This is [not documented well](https://github.com/pytorch/pytorch/issues/58489), but seems to be intentional.

## Number of elements

Reference: [#58741](https://github.com/pytorch/pytorch/issues/58741)

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/array_object.html#size) that the `size` attribute of an array should return an integer with the number of elements of the array.

In PyTorch the `size` attribute is a method that returns the [`shape`](https://data-apis.org/array-api/latest/API_specification/array_object.html#shape) when called. The number of elements is accessible through the `numel()` method:

```python
import torch

t = torch.empty((2, 3, 4))
size = 2 * 3 * 4

assert t.size() == t.shape
assert t.numel() == size
```

## Missing operators

Reference: [#58742](https://github.com/pytorch/pytorch/issues/58742)

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/array_object.html#size) the following operators, but they are not supported by PyTorch:

```shell
pytest xptests/test_signatures.py::test_has_names | grep FAILED | sed -E 's#FAILED xptests/test_signatures.py::test_has_names\[([a-zA-Z_\.]+)\].*#- \`\1\`#'
```

- `__array_namespace__`
- `to_device`
- `__imatmul__`
- `astype`
- `broadcast_arrays`
- `bitwise_invert`
- `matrix_transpose`
- `vecdot`
- `expand_dims`
- `permute_dims`
- `unique_all`
- `unique_counts`
- `unique_inverse`
- `unique_values`
- `linalg.diagonal`
- `linalg.matrix_transpose`
- `linalg.outer`
- `linalg.tensordot`
- `linalg.trace`
- `linalg.vecdot`

In addition `__ipow__` needs to be implemented, but will not be picked up by this test, since [it exists as a stub](https://github.com/pytorch/pytorch/blob/dc67b47bc9d53dbeb898a4d920b0225ac73629ec/torch/tensor.py#L531-L536) for `__torch_function__` overrides.

## Negative step sizes for slicing

Reference: [#59786](https://github.com/pytorch/pytorch/issues/59786)

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/indexing.html#slice-syntax) that the slicing syntax needs to be equivalent to Pythons [`slice()`](https://docs.python.org/3/library/functions.html#slice) syntax. That includes negative step sizes that PyTorch does not support:

```python
import torch

t = torch.rand((3,))
t[::-1]
```

```
ValueError: step must be greater than zero
```

## Single ellipsis for slicing

Reference: [#59787](https://github.com/pytorch/pytorch/issues/59787)

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/indexing.html#multi-axis-indexing) that the slice for nd tensors shall contain at most a single [ellipsis](https://docs.python.org/3/library/constants.html#Ellipsis) (`...`). PyTorch allows an arbitrary number.

```
import torch

a = torch.rand((1, 2, 3, 4, 5))
a[..., 0, ..., ...]
```

## `torch.asarray` does not detect dtype of Python scalars

Reference: [#70591](https://github.com/pytorch/pytorch/issues/70591)

```python
import torch

for obj in (True, 1, 1.0):
    print(f"{obj} -> {torch.asarray(obj).dtype}")
```

```
True -> torch.float32
1 -> torch.float32
1.0 -> torch.float32
```

## Bitwise shifts should retain the same dtype as the first input

Reference: [#59867](https://github.com/pytorch/pytorch/issues/59867)

The array API specification stipulates that [`bitwise_left_shift`](https://data-apis.org/array-api/latest/API_specification/elementwise_functions.html#id29) and [`bitwise_left_shift`](https://data-apis.org/array-api/latest/API_specification/elementwise_functions.html#id35) (as well as the corresponding magics `__lshift__` and `__rshift__`) do not participate in the default type promotion, but rather retain the dtype of the first input. PyTorch performs the standard type promotion.

```python
>>> x1 = torch.tensor([0, 1], dtype=torch.uint8)
>>> x2 = torch.tensor([0, 1], dtype=torch.int8)
>>> (x1 << x2).dtype, x1.dtype
(torch.int16, torch.uint8)
>>> (x2 << x1).dtype, x2.dtype
(torch.int16, torch.int8)
>>> (x1 >> x2).dtype, x1.dtype
(torch.int16, torch.uint8)
>>> (x2 >> x1).dtype, x2.dtype
(torch.int16, torch.int8)
```

## `torch.(min|max)(..., dim=...)` diverges from array API specification

Reference: [#58745](https://github.com/pytorch/pytorch/issues/58745)

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/statistical_functions.html?highlight=max#max-x-axis-none-keepdims-false) that `torch.(max|min)` must always return a tensor. [If called with `dim=...`](https://github.com/pytorch/pytorch/issues/58745#issuecomment-846921844)  `torch.return_types.(max|min)` is returned instead.


## Bitwise shifts are broken in some cases if shift is greater than bit size

Reference: [#70904](https://github.com/pytorch/pytorch/issues/70904)

Left- or right-shifting the bits of an integer tensor by equal or more than the number of bits of the dtype, should always result in `0`. This is not the case for some shifts:

```python
import torch

dtype = torch.uint8
info = torch.iinfo(dtype)
input = torch.tensor(info.max, dtype=dtype)

for other in range(info.bits, 100):
    other = torch.full_like(input, other)

    result_left = input << other
    if result_left != 0:
        print(f"{int(input)} << {int(other)}: {int(result_left)}")

    result_right = input >> other
    if result_right != 0:
        print(f"{int(input)} >> {int(other)}: {int(result_right)}")
```

```
255 << 32: 255
255 >> 32: 255
255 << 33: 254
255 >> 33: 127
255 << 34: 252
255 >> 34: 63
255 << 35: 248
255 >> 35: 31
255 << 36: 240
255 >> 36: 15
255 << 37: 224
255 >> 37: 7
255 << 38: 192
255 >> 38: 3
255 << 39: 128
255 >> 39: 1
255 << 64: 255
255 >> 64: 255
255 << 65: 254
255 >> 65: 127
255 << 66: 252
255 >> 66: 63
255 << 67: 248
255 >> 67: 31
255 << 68: 240
255 >> 68: 15
255 << 69: 224
255 >> 69: 7
255 << 70: 192
255 >> 70: 3
255 << 71: 128
255 >> 71: 1
255 << 96: 255
255 >> 96: 255
255 << 97: 254
255 >> 97: 127
255 << 98: 252
255 >> 98: 63
255 << 99: 248
255 >> 99: 31
```


## Python scalars should be promoted to the same `dtype` as the respective tensor

Reference: [#59868](https://github.com/pytorch/pytorch/issues/59868)

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/type_promotion.html#mixing-arrays-with-python-scalars) that for binary operations involving a tensor and a Python scalar, the scalar needs to be converted to the same `dtype` as the tensor before the operation is performed. PyTorch casts the scalar to a tensor based on its `dtype` and afterwards performs the default type promotion for the operator. 

This can lead to overflows if the tensor `dtype` can hold the values of the scalar, but the automatically determined `dtype` cannot.

```python
>>> torch.tensor(0, dtype=torch.float32) + 2.0 ** 63
tensor(9.2234e+18)
>>> torch.tensor(0, dtype=torch.float32) + 2 ** 63
RuntimeError: Overflow when unpacking long
```

## `astype` / `to_device` 

The array API specification stipulates that an array object (read `torch.Tensor`) should have a [`.astype()`](https://data-apis.org/array-api/latest/API_specification/data_type_functions.html#astype-x-dtype-copy-true) and [`.to_device()`](https://data-apis.org/array-api/latest/API_specification/array_object.html?highlight=to_device#to-device-self-device-stream-none) method. Both functionalities are currently performed by [`torch.Tensor.to`](https://pytorch.org/docs/master/generated/torch.Tensor.to.html#torch-tensor-to).

## `full` should take an integer size

Reference: [#70906](https://github.com/pytorch/pytorch/issues/70906)

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/creation_functions.html?highlight=full#full-shape-fill-value-dtype-none-device-none) that the `size` parameter can also be an `int`, which should be treated the same as a sequence of `int`'s with a single element.

## `finfo(...).tiny` should be aliased to `finfo(...).smallest_normal`

Reference: [#70909](https://github.com/pytorch/pytorch/issues/70909)

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/data_type_functions.html?highlight=smallest_normal#finfo-type) that the

> smallest positive floating-point number with full precision

should be called `smallest_normal`. We are currently calling it [`tiny`](https://pytorch.org/docs/master/type_info.html#torch.torch.finfo). See [this discussion](https://github.com/data-apis/array-api/pull/129#discussion_r577993658) for the motivation behind this.

## `eye` should support other diagonals than the main one

Reference: [#70910](https://github.com/pytorch/pytorch/issues/70910)

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/creation_functions.html#eye-n-rows-n-cols-none-k-0-dtype-none-device-none) that the diagonal does not have to be the main one. This is controlled by the `k` parameter. Positive values correspond to upper diagonals whereas negative values correspond to lower diagonals. By default `k=0` which is the main diagonal and thus our current behavior.

## support setting `step` in `arange` without setting `end`

Reference: [#70914](https://github.com/pytorch/pytorch/issues/70914)

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/creation_functions.html#arange-start-stop-none-step-1-dtype-none-device-none) that only the first argument in `arange` is positional only. The other two, `end` and `step` need to be independently usable.

```python
>>> torch.arange(5, end=10)
tensor([5, 6, 7, 8, 9])
>>> torch.arange(5, step=2)
TypeError: arange() received an invalid combination of arguments - got (int, step=int), but expected one of:
 * (Number end, *, Tensor out, torch.dtype dtype, torch.layout layout, torch.device device, bool pin_memory, bool requires_grad)
 * (Number start, Number end, Number step, *, Tensor out, torch.dtype dtype, torch.layout layout, torch.device device, bool pin_memory, bool requires_grad)
```

## `arange` should return empty array if bounds are inconsistent with step sign

Reference: [#70915](https://github.com/pytorch/pytorch/issues/70915)

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/creation_functions.html#returns) that the following should hold for the return value of `torch.arange`:

> The length of the output array must be `ceil((stop-start)/step)` if `stop - start` and `step` have the same sign, and length `0` otherwise. We currently bail out:

```python
>>> torch.arange(1, 0)
RuntimeError: upper bound and larger bound inconsistent with step sign
```

## uint8 scalar tensors cannot be used for integer indexing

Reference: [#70916](https://github.com/pytorch/pytorch/issues/70916)

Integer, scalar tensors should behave like integers when used as index. Tensors of dtype `torch.uint8` deviate from that:

```python
import torch

t_1d_single = torch.empty(1)
t_1d_multi = torch.empty(2)

for dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
    print("single", dtype, t_1d_single[torch.tensor(0, dtype=dtype)].shape)
    print("multi1", dtype, t_1d_multi[torch.tensor(0, dtype=dtype)].shape)
    print("multi2", dtype, t_1d_multi[torch.tensor(1, dtype=dtype)].shape)
    print("#" * 50)
```

```
single torch.uint8 torch.Size([0, 1])
multi1 torch.uint8 torch.Size([0, 2])
multi2 torch.uint8 torch.Size([1, 2])
##################################################
single torch.int8 torch.Size([])
multi1 torch.int8 torch.Size([])
multi2 torch.int8 torch.Size([])
##################################################
single torch.int16 torch.Size([])
multi1 torch.int16 torch.Size([])
multi2 torch.int16 torch.Size([])
##################################################
single torch.int32 torch.Size([])
multi1 torch.int32 torch.Size([])
multi2 torch.int32 torch.Size([])
##################################################
single torch.int64 torch.Size([])
multi1 torch.int64 torch.Size([])
multi2 torch.int64 torch.Size([])
##################################################
```

## make `torch.(ceil|floor|round|trunc) no-ops for integer inputs

Reference: [#70918](https://github.com/pytorch/pytorch/issues/70918)

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/elementwise_functions.html?highlight=bitwise_left_shift#id38) that `torch.(ceil|floor|round|trunc)` should be no-ops for integer inputs. We currently have no functionality implemented for these cases

```python
>>> t = torch.tensor(0)
>>> torch.ceil(t)
RuntimeError: "ceil" "_vml_cpu" not implemented for 'Long'
>>> torch.floor(t)
RuntimeError: "floor" "_vml_cpu" not implemented for 'Long'
>>> torch.round(t)
RuntimeError: "round" "_vml_cpu" not implemented for 'Long'
>>> torch.trunc(t)
RuntimeError: "trunc" "_vml_cpu" not implemented for 'Long'
```

## `linspace` should support an `endpoint` parameter

Reference: [#70919](https://github.com/pytorch/pytorch/issues/70919)

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/creation_functions.html#linspace-start-stop-num-dtype-none-device-none-endpoint-true) that `torch.linspace` should have a `endpoint: bool = True` parameter. If `False`, the right bound should be exclusive, i.e. `[start, end)` rather than `[start, end]`.

## `unique` should be split into four partial functions

Reference: [#70920](https://github.com/pytorch/pytorch/issues/70920)

The array API specification stipulates that we should have four partial functions for uniqueness computation rather than just one:

- [`unique_values`](https://data-apis.org/array-api/latest/API_specification/set_functions.html#unique-values-x): `torch.unique`
- [`unique_counts`](https://data-apis.org/array-api/latest/API_specification/set_functions.html#unique-counts-x): `partial(torch.unique, return_counts=True)`[^1]
- [`unique_inverse`](https://data-apis.org/array-api/latest/API_specification/set_functions.html#unique-inverse-x): `partial(torch.unique, return_inverse=True)`[^1]
- [`unique_all`](https://data-apis.org/array-api/latest/API_specification/set_functions.html#unique-all-x): roughly `partial(torch.unique, return_inverse=True, return_counts=True)`[^1], with the caveat that actually 4 tensors should be returned, with the second element being the indices that denote the first occurences of the values in the input.

[^1]: We currently return plain tuples, but for compliance with the array API, we need to return named tuples.

## `sort` should only return the sorted input

Reference: [#70921](https://github.com/pytorch/pytorch/issues/70921)

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/generated/signatures.sorting_functions.sort.html) that `torch.sort` should only return the sorted input. Currently, we return a named tuple where the `.values` field corresponds to the specification of the array API.

## `argsort` is missing the `stable` parameter

Reference: [#70922](https://github.com/pytorch/pytorch/issues/70922)

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/generated/signatures.sorting_functions.argsort.html) that `torch.argsort` should support the same parameters as `torch.argsort`. Excluding `out`, `stable: bool = True` is missing from `torch.argsort`.

## type promotion is broken in `torch.where`

Reference: [#70923](https://github.com/pytorch/pytorch/issues/70923)

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/searching_functions.html?highlight=where#id7) that the return value of `torch.where` should undergo regular type promotion. Currently we do not support different dtypes for `x` and `y`:

```python
import torch

condition = torch.tensor([False, True])
x = torch.ones(2, dtype=torch.float32)
y = torch.zeros(2, dtype=torch.float64)

torch.where(condition, x, y)
```

```
RuntimeError: expected scalar type float but found double
```

Note that the error message is also misleading since we deal with 1d tensors here. 

## change supported arguments for parameter `dim` in `squeeze`

Reference: [#70923](https://github.com/pytorch/pytorch/issues/70923)

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/manipulation_functions.html?highlight=squeeze#id11) that

1. `dim` should accept a tuple of `int`'s indicating all dimensions that should be squeezed.
2. `dim` should have no default value. Currently, not passing anything will squeeze all singleton dimension.

## change supported arguments for parameter `dim` in `concat`

Reference: [#70923](https://github.com/pytorch/pytorch/issues/70923)

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/manipulation_functions.html?highlight=concat#parameters) that

1. `dim=None` flattens all inputs before concatenating.
2. negative `dim`'s index the dimensions starting from the last one.

## axis to dim remapping is not working for flip and roll

Reference: [#71210](https://github.com/pytorch/pytorch/issues/71210)

Although not documented, the Python interface of PyTorch ops internally maps `axis` keywords to `dim` for `numpy` and in turn also for array API compliance:

```python
>>> t = torch.empty(1, 2, 1, 3)
>>> t.squeeze(dim=2).shape
torch.Size([1, 2, 3])
>>> t.squeeze(axis=2).shape
torch.Size([1, 2, 3])
```

This mapping does not work for the following ops:

| op     | array API parameter | PyTorch parameter |
|--------| --- | --- |
| `flip` | [`axis`](https://data-apis.org/array-api/latest/API_specification/manipulation_functions.html#id3) | [`dims`](https://pytorch.org/docs/master/generated/torch.flip.html) |
| `roll` | [`axis`](https://data-apis.org/array-api/latest/API_specification/manipulation_functions.html#roll-x-shift-axis-none) | [`dims`](https://pytorch.org/docs/master/generated/torch.roll.html) |

## support setting `keepdim` without setting `dim`

Reference: [#71209](https://github.com/pytorch/pytorch/issues/71209)

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/utility_functions.html#all-x-axis-none-keepdims-false) that only the first argument in `all` is positional only. The other two, `dim` and `keepdim` need to be independently usable.

```python
>>> t = torch.full((2, 3), True)
>>> torch.all(t)
tensor(True)
>>> torch.all(t, dim=1)
tensor([True, True])
>>> torch.all(t, dim=1, keepdims=True)
tensor([[True],
        [True]])
>>> torch.all(t, keepdims=True)
TypeError: all() received an invalid combination of arguments - got (Tensor, keepdims=bool), but expected one of:
 * (Tensor input, *, Tensor out)
 * (Tensor input, int dim, bool keepdim, *, Tensor out)
 * (Tensor input, name dim, bool keepdim, *, Tensor out)
```

The same applies to the following other ops:

- `any`
- `max`
- `mean`
- `prod`
- `std`
- `sum`
- `var`

This issue is related to #70914, which likely uses the same underlying functionality to parse inputs.
