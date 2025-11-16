# Patterns and Practices in PyTorch

**Analysis based on commit:** [`5d99a795f54d6bf14e39ae12df58d760d4fd8984`](https://github.com/pytorch/pytorch/tree/5d99a795f54d6bf14e39ae12df58d760d4fd8984)

## Introduction

Building a framework as complex as PyTorch requires not just excellent architecture, but also robust development practices. PyTorch supports 15+ backends (CPU, CUDA, MPS, XPU, etc.), 20+ data types, and thousands of operators—all while maintaining high code quality, comprehensive testing, and clear documentation.

In this post, we'll explore the patterns and practices that make PyTorch development scalable:
- How tests are organized to cover the massive combinatorial space of devices × dtypes × operators
- Code quality infrastructure that catches bugs before they reach production
- Documentation generation that keeps API references in sync with implementation
- Design patterns that appear throughout the codebase

Understanding these practices is invaluable for anyone contributing to PyTorch or building similar large-scale systems.

## Objectives

By the end of this post, you'll understand:

1. PyTorch's sophisticated test parametrization system
2. Code quality tools and linting infrastructure
3. Documentation generation with Sphinx
4. Common design patterns throughout the codebase
5. CI/CD workflows and quality gates

## Testing Infrastructure

### The Combinatorial Explosion Problem

Testing PyTorch is challenging because of the combinatorial explosion:

```
# Number of test cases:
~1000 operators ×
  20+ data types (float32, int64, complex128, etc.) ×
  15+ devices (CPU, CUDA, MPS, XPU, etc.) ×
  multiple memory layouts (contiguous, channels_last, etc.) ×
  multiple use cases (requires_grad, sparse, etc.)
= Millions of potential test cases
```

Writing each test manually is infeasible. PyTorch solves this with **device-type parametrization**.

### Device-Type Test Instantiation

**Source:** [`torch/testing/_internal/common_device_type.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/testing/_internal/common_device_type.py)

The pattern:

```python
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, dtypes
)
from torch.testing._internal.common_utils import TestCase

class TestComplexTensor(TestCase):
    # Write ONE test that works for ANY device
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_to_list(self, device, dtype):
        # device parameter is injected by framework
        tensor = torch.zeros((2, 2), device=device, dtype=dtype)
        result = tensor.tolist()
        self.assertEqual(result, [[0, 0], [0, 0]])

# Magic happens here:
instantiate_device_type_tests(TestComplexTensor, globals())

if __name__ == "__main__":
    run_tests()
```

**What `instantiate_device_type_tests()` does:**

1. **Removes the original class** from globals
2. **Creates specialized classes** for each device:
   - `TestComplexTensorCPU`
   - `TestComplexTensorCUDA`
   - `TestComplexTensorMPS`
   - `TestComplexTensorXPU`
   - etc.
3. **Parametrizes each test** with appropriate dtypes
4. **Adds each specialized class** to globals for pytest discovery

**Result:** One test method becomes dozens of test cases automatically:
```
TestComplexTensorCPU::test_to_list_float32_cpu
TestComplexTensorCPU::test_to_list_float64_cpu
TestComplexTensorCPU::test_to_list_complex64_cpu
TestComplexTensorCUDA::test_to_list_float32_cuda
TestComplexTensorCUDA::test_to_list_float64_cuda
# ... etc.
```

See the implementation: [`torch/testing/_internal/common_device_type.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/testing/_internal/common_device_type.py#L500-L600).

### Test Decorators for Fine-Grained Control

```python
from torch.testing._internal.common_device_type import (
    onlyCPU, onlyCUDA, skipIfRocm, precisionOverride, toleranceOverride
)

class TestOperators(TestCase):
    # Only run on CPU
    @onlyCPU
    def test_cpu_only_feature(self, device):
        pass

    # Skip on ROCm (AMD GPUs)
    @skipIfRocm
    def test_incompatible_with_rocm(self, device):
        pass

    # Override precision for numerical comparisons
    @precisionOverride({torch.float16: 1e-2, torch.bfloat16: 1e-2})
    def test_with_low_precision(self, device, dtype):
        pass

    # Different tolerance per dtype
    @toleranceOverride({torch.float64: tol(atol=1e-7, rtol=1e-5)})
    def test_high_precision(self, device, dtype):
        pass
```

### Testing Utilities

**Tensor Creation:**

```python
# torch/testing/_creation.py
from torch.testing import make_tensor

# Creates test tensors with smart defaults
x = make_tensor(
    (3, 4, 5),
    dtype=torch.float32,
    device='cuda',
    low=-9,                # Range for random values
    high=9,
    requires_grad=True,
    noncontiguous=True,    # Creates non-contiguous tensor
    exclude_zero=False,    # Whether to exclude zeros
)
```

**Tensor Comparison:**

```python
# torch/testing/_comparison.py
from torch.testing._internal.common_utils import TestCase

class MyTest(TestCase):
    def test_operation(self):
        result = my_operation(input)
        expected = ground_truth(input)

        # Smart comparison with dtype-appropriate tolerances
        self.assertEqual(result, expected)
        # For float32: rtol=1.3e-6, atol=1e-5
        # For float64: rtol=1e-7, atol=1e-7
        # For float16: rtol=1e-3, atol=1e-5
```

The `assertEqual` method automatically:
- Checks shapes match
- Applies dtype-appropriate tolerances
- Provides detailed error messages showing where tensors differ
- Handles special values (NaN, Inf) correctly

### OpInfo: Declarative Operator Testing

For systematic operator testing, PyTorch uses `OpInfo` definitions:

**Source:** [`torch/testing/_internal/common_methods_invocations.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/testing/_internal/common_methods_invocations.py)

```python
op_db = [
    OpInfo(
        'add',
        dtypes=all_types_and_complex(),
        sample_inputs_func=sample_inputs_add,
        supports_autograd=True,
        supports_gradgrad=True,
        supports_forward_ad=True,
        skips=(
            # Skip certain dtype/device combinations
            DecorateInfo(unittest.skip("Known issue"), 'TestGradients', 'test_fn_grad'),
        ),
    ),
    # ... hundreds more
]
```

Generic tests consume `OpInfo` to automatically test:
- Forward computation
- Backward pass (gradients)
- Second derivatives (gradgrad)
- JIT compilation
- Serialization/deserialization
- TorchScript export

## Code Quality Infrastructure

### Linting with Multiple Tools

PyTorch uses a **lint orchestrator** to run multiple linters:

**Source:** [`.lintrunner.toml`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/.lintrunner.toml)

```toml
# Linters:
# 1. Flake8 - Python style and common bugs
# 2. Clang-Format - C++ formatting
# 3. Pyrefly - Advanced Python type checking
# 4. Others: mypy, ruff, isort
```

### Flake8 Configuration

**Source:** [`.flake8`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/.flake8)

```ini
[flake8]
max-line-length = 120

# Enabled plugins:
# - flake8-bugbear (B): Logic errors and anti-patterns
# - flake8-simplify (SIM): Code simplification suggestions
# - torchfix (TOR): PyTorch-specific linting
# - flake8-comprehensions (C4): Comprehension optimizations

# Example rules:
# B007: Loop variable not used
# B905: zip() without strict parameter
# SIM108: Use ternary operator instead of if/else
# TOR001: torch.nn.functional.* used without F alias
# TOR101: Unnecessary reshape before view
```

**TorchFix** is particularly interesting—it's a PyTorch-specific linter that catches issues like:
```python
# Bad: inefficient
x = x.reshape(10, 10).view(-1)  # TOR101

# Good:
x = x.reshape(-1)

# Bad: no alias
import torch.nn.functional
result = torch.nn.functional.relu(x)  # TOR001

# Good:
import torch.nn.functional as F
result = F.relu(x)
```

### Type Checking with MyPy

**Source:** [`mypy.ini`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/mypy.ini)

```ini
[mypy]
python_version = 3.11
check_untyped_defs = True
disallow_untyped_defs = True  # Require type annotations
warn_redundant_casts = True

# Gradually typing the codebase:
files =
    torch,
    caffe2,
    test/test_complex.py,
    test/test_torch.py,
    # Selected files with type annotations
```

**Example typed code:**

```python
# torch/nn/modules/linear.py
from typing import Optional
import torch
from torch import Tensor

class Linear(Module):
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        ...

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)
```

### Import Sorting

**Source:** [`pyproject.toml`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/pyproject.toml)

```toml
[tool.isort]
profile = "black"
line_length = 88
multi_line_output = 3
include_trailing_comma = true

# Enforces consistent import ordering:
# 1. Standard library
# 2. Third-party packages
# 3. PyTorch imports
# 4. Local imports
```

Example:
```python
# Standard library
import os
import sys
from typing import Optional

# Third-party
import numpy as np

# PyTorch
import torch
from torch import Tensor

# Local
from .utils import helper_function
```

## Documentation Infrastructure

### Sphinx Configuration

PyTorch uses **Sphinx** for documentation generation:

**Source:** [`docs/source/conf.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/docs/source/conf.py)

```python
extensions = [
    'sphinx.ext.autodoc',           # Auto-generate docs from docstrings
    'sphinx.ext.autosummary',       # Auto-generate summary tables
    'sphinx.ext.napoleon',          # Google/NumPy docstring support
    'sphinxcontrib.katex',          # LaTeX math rendering
    'sphinx_copybutton',            # Copy code button
    'myst_nb',                      # Jupyter notebook support
    'sphinx.ext.linkcode',          # Link to source code
]

autosummary_generate = True
html_theme = 'pytorch_sphinx_theme'
```

### Docstring Conventions

PyTorch follows NumPy/Google docstring style:

```python
def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This function supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        input: input tensor of shape :math:`(*, H_\text{in})`
        weight: weight tensor of shape :math:`(\text{out\_features}, \text{in\_features})`
        bias: optional bias tensor of shape :math:`(\text{out\_features})`. Default: ``None``

    Shape:
        - Input: :math:`(*, H_\text{in})` where :math:`*` means any number of dimensions
          and :math:`H_\text{in} = \text{in\_features}`
        - Weight: :math:`(\text{out\_features}, \text{in\_features})`
        - Bias: :math:`(\text{out\_features})`
        - Output: :math:`(*, H_\text{out})` where :math:`H_\text{out} = \text{out\_features}`

    Examples::

        >>> input = torch.randn(128, 20)
        >>> weight = torch.randn(30, 20)
        >>> output = F.linear(input, weight)
        >>> output.size()
        torch.Size([128, 30])
    """
```

**Key elements:**
- **Raw strings** (`r"""...""""`) for LaTeX math
- **Math notation:** `:math:\`...\``
- **Cross-references:** `:ref:\`...\``, `:func:\`...\``
- **Shape documentation:** Explicit input/output shapes
- **Examples:** Runnable code snippets
- **Notes/Warnings:** Additional context

### API Documentation Generation

From docstrings to web documentation:

```
Python source with docstrings
    ↓
sphinx-build reads via autodoc
    ↓
Generates .rst files via autosummary
    ↓
Renders HTML with pytorch_sphinx_theme
    ↓
Published to pytorch.org/docs
```

Cross-references are resolved automatically:
```python
"""See :func:`torch.add` for element-wise addition."""
# Becomes a hyperlink to torch.add documentation
```

## Design Patterns Throughout PyTorch

### 1. Builder Pattern: TensorIterator

```cpp
// aten/src/ATen/TensorIterator.h
auto iter = TensorIteratorConfig()
    .add_output(out)
    .add_input(input1)
    .add_input(input2)
    .promote_inputs_to_common_dtype(true)
    .check_all_same_dtype(false)
    .build();
```

**Benefits:**
- Readable configuration
- Optional parameters don't clutter constructor
- Validation happens in `build()`

### 2. Dispatch Stub Pattern

**Source:** [`aten/src/ATen/native/DispatchStub.h`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/native/DispatchStub.h)

```cpp
// Define stub (function pointer)
using add_fn = void(*)(TensorIterator&, const Scalar&);
DECLARE_DISPATCH(add_fn, add_stub);

// Register CPU implementation
REGISTER_ARCH_DISPATCH(add_stub, DEFAULT, &add_kernel_cpu);
REGISTER_ARCH_DISPATCH(add_stub, AVX2, &add_kernel_avx2);
REGISTER_ARCH_DISPATCH(add_stub, AVX512, &add_kernel_avx512);

// Use:
add_stub(iter.device_type(), iter, alpha);
// Automatically selects AVX512 kernel on Skylake-X CPUs
```

**Benefits:**
- Runtime CPU feature detection
- Easy to add optimized kernels for new CPUs
- No runtime overhead (direct function pointer call)

### 3. RAII for CUDA Context Management

```cpp
// c10/cuda/CUDAGuard.h
{
  at::cuda::CUDAGuard guard(device_id);
  // CUDA operations happen on device_id
  kernel<<<blocks, threads>>>(args);
}  // Automatically restores previous device
```

### 4. Intrusive Pointer Pattern

```cpp
// c10/util/intrusive_ptr.h
c10::intrusive_ptr<TensorImpl> ptr(new TensorImpl());
// Reference count stored IN the TensorImpl object
// More cache-friendly than std::shared_ptr
```

### 5. Registry Pattern

```cpp
// torch/csrc/jit/passes/pass_manager.cpp
struct PassRegistry {
  std::unordered_map<std::string, PassFunction> passes_;

  void registerPass(const std::string& name, PassFunction fn) {
    passes_[name] = fn;
  }

  void runPass(const std::string& name, Graph& graph) {
    passes_.at(name)(graph);
  }
};
```

Used for:
- Operator registration
- Optimization passes
- Backend registration
- Custom operator registration

## CI/CD Workflows

### GitHub Actions Structure

**Source:** [`.github/workflows/`](https://github.com/pytorch/pytorch/tree/5d99a795f54d6bf14e39ae12df58d760d4fd8984/.github/workflows)

PyTorch uses **reusable workflows**:

```yaml
# _linux-test.yml (reusable workflow)
name: linux-test
on:
  workflow_call:
    inputs:
      build-environment: { required: true, type: string }
      test-matrix: { required: true, type: string }
      docker-image: { required: true, type: string }

jobs:
  test:
    strategy:
      matrix: ${{ fromJSON(inputs.test-matrix) }}
    runs-on: ${{ matrix.runner }}
    steps:
      - name: Run tests
        run: |
          python test/run_test.py --verbose
```

Called from main workflow:
```yaml
# pull.yml (main workflow)
jobs:
  linux-test:
    uses: ./.github/workflows/_linux-test.yml
    with:
      build-environment: "linux-focal-py3.9-gcc7"
      test-matrix: ${{ needs.generate-matrix.outputs.matrix }}
      docker-image: "pytorch/pytorch:latest"
```

**Benefits:**
- Reusable across Linux, macOS, Windows
- Easy to add new platforms
- Consistent test execution

### Test Sharding

Large test files are sharded across multiple runners:

```python
# test/run_test.py
pytest test/test_nn.py --shard-id=1 --num-shards=4  # Run 1/4 of tests
pytest test/test_nn.py --shard-id=2 --num-shards=4  # Run 2/4 of tests
# ...
```

Custom sharding logic: [`test/pytest_shard_custom.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/test/pytest_shard_custom.py)

### Quality Gates

Before merging, code must pass:

1. **Lint checks:** Flake8, clang-format, mypy, ruff
2. **Unit tests:** All device types (CPU, CUDA, MPS, XPU, etc.)
3. **Integration tests:** Multi-GPU, distributed training
4. **Performance tests:** Benchmark against baseline
5. **Documentation build:** Sphinx documentation compiles
6. **Type checking:** MyPy passes on typed modules

## Error Handling Patterns

### TORCH_CHECK for Input Validation

```cpp
// aten/src/ATen/native/Linear.cpp
Tensor linear(const Tensor& input, const Tensor& weight, const optional<Tensor>& bias) {
  TORCH_CHECK(input.dim() >= 2, "Expected input to be at least 2D, got ", input.dim(), "D");
  TORCH_CHECK(weight.dim() == 2, "Expected weight to be 2D, got ", weight.dim(), "D");

  if (bias.has_value()) {
    TORCH_CHECK(bias->dim() == 1, "Expected bias to be 1D, got ", bias->dim(), "D");
  }

  // Implementation...
}
```

**Benefits:**
- Clear error messages
- Early validation
- Consistent error format

### TORCH_INTERNAL_ASSERT for Invariants

```cpp
void internal_function(const Tensor& t) {
  TORCH_INTERNAL_ASSERT(t.defined(), "Tensor must be defined");
  // Should never happen in correct code
  // If it does, it's a PyTorch bug, not user error
}
```

## Key Takeaways

1. **Test parametrization is crucial:** Device-type instantiation enables testing across the combinatorial space without code duplication.

2. **Multiple linters provide defense in depth:** Flake8 catches style issues, TorchFix catches PyTorch-specific problems, MyPy ensures type safety.

3. **Documentation is code:** Docstrings are the source of truth, automatically rendered to web documentation with Sphinx.

4. **Design patterns enable scalability:** Builder, Registry, DispatchStub, and RAII patterns appear throughout the codebase for good reason.

5. **CI/CD must be fast and reliable:** Reusable workflows, test sharding, and quality gates ensure code quality without slowing development.

6. **Type annotations are gradually adopted:** PyTorch is progressively adding type hints, file by file, to improve IDE support and catch bugs earlier.

## What's Next

In the next post, we'll explore how to extend PyTorch:
- Writing custom operators with C++ and CUDA
- Registering operators with the dispatcher
- Creating custom autograd functions
- Building PyTorch extensions and plugins
- Integration patterns with external libraries

## References

- **Device-Type Testing:** [`torch/testing/_internal/common_device_type.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/testing/_internal/common_device_type.py)
- **OpInfo Database:** [`torch/testing/_internal/common_methods_invocations.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/torch/testing/_internal/common_methods_invocations.py)
- **Linting Config:** [`.lintrunner.toml`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/.lintrunner.toml)
- **Sphinx Config:** [`docs/source/conf.py`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/docs/source/conf.py)
- **CI Workflows:** [`.github/workflows/`](https://github.com/pytorch/pytorch/tree/5d99a795f54d6bf14e39ae12df58d760d4fd8984/.github/workflows)
- **DispatchStub:** [`aten/src/ATen/native/DispatchStub.h`](https://github.com/pytorch/pytorch/blob/5d99a795f54d6bf14e39ae12df58d760d4fd8984/aten/src/ATen/native/DispatchStub.h)

---

*This analysis is based on PyTorch commit [`5d99a79`](https://github.com/pytorch/pytorch/tree/5d99a795f54d6bf14e39ae12df58d760d4fd8984).*
