# Array API Standard Test Suite for PyTorch

This documents all deviations of [PyTorch](https://pytorch.org) from the array API specification identified by the test suite.

Array API specification commit: [64cad4b](https://github.com/data-apis/array-api/tree/64cad4b95cc2f60d89c82b88b3e3e91b90d4a4ae)

PyTorch version: `1.10.0.dev20210609`

## Unsigned integer data types

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/data_types.html) four unsigned integer data types:

1. `uint8`,
2. `uint16`,
3. `uint32`, and
4. `uint64`.

Of those, PyTorch [only supports `uint8`](https://pytorch.org/docs/stable/tensor_attributes.html#torch-dtype).

Reference: [#58734](https://github.com/pytorch/pytorch/issues/58734)

## Inter-category type promotion involving 0d-array's

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

Reference: [#58736](https://github.com/pytorch/pytorch/issues/58736)

## Constants

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/constants.html) four numeric constants:

- `e`
- `inf`
- `nan`
- `pi`

PyTorch supports none.

Reference: [#58739](https://github.com/pytorch/pytorch/issues/58739)

## `logical_*` operators with scalars

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/constants.html) that the operators `logical_(and|not|or|xor)` should accept one scalar the same way arithmetic functions do.

PyTorch does not support that, although the magic methods seem to work just fine:

```python
import torch

a = torch.tensor(True)
b = True

assert a & b
assert torch.logical_and(a, b)
```

```
tensor(False)
TypeError: logical_and(): argument 'other' (position 2) must be Tensor, not bool
```

Reference: [#58740](https://github.com/pytorch/pytorch/issues/58740)

## Number of elements

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/array_object.html#size) that the `size` attribute of an array should return an integer with the number of elements of the array.

In PyTorch the `size` attribute is a method that returns the [`shape`](https://data-apis.org/array-api/latest/API_specification/array_object.html#shape) when called. The number of elements is accessible through the `numel()` method:

```python
import torch

t = torch.empty((2, 3, 4))
size = 2 * 3 * 4

assert t.size() == t.shape
assert t.numel() == size
```

Reference: [#58741](https://github.com/pytorch/pytorch/issues/58741)

## Missing operators

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/array_object.html#size) the following operators, but they are not supported by PyTorch:

```shell
pytest array_api_tests/test_signatures.py::test_has_names \
    | grep FAILED \
    | sed -E "s/FAILED array_api_tests\/test_signatures.py::test_has_names\[(\w+)\].*/- \`\1\`/"
```

- `__array_namespace__`
- `__dlpack__`
- `__dlpack_device__`
- `__rand__`
- `__rlshift__`
- `__imatmul__`
- `__ror__`
- `__rrshift__`
- `__rxor__`
- `from_dlpack`
- `broadcast_arrays`
- `bitwise_left_shift`
- `bitwise_invert`
- `bitwise_right_shift`
- `vecdot`
- `concat`
- `expand_dims`

Reference: [#58742](https://github.com/pytorch/pytorch/issues/58742)

## Negative step sizes for slicing

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/indexing.html#slice-syntax) that the slicing syntax needs to be equivalent to Pythons [`slice()`](https://docs.python.org/3/library/functions.html#slice) syntax. That includes negative step sizes that PyTorch does not support:

```python
import torch

t = torch.rand((3,))
t[::-1]
```

```
ValueError: step must be greater than zero
```

Reference: [#59786](https://github.com/pytorch/pytorch/issues/59786)

## Single ellipsis for slicing

The [array API specification stipulates](https://data-apis.org/array-api/latest/API_specification/indexing.html#multi-axis-indexing) that the slice for nd tensors shall contain at most a single [ellipsis](https://docs.python.org/3/library/constants.html#Ellipsis) (`...`). PyTorch allows an arbitrary number.

```
import torch

a = torch.rand((1, 2, 3, 4, 5))
a[..., 0, ..., ...]
```

Reference: [#59787](https://github.com/pytorch/pytorch/issues/59787)