# OpInfo V2 User Guide

## Overview

OpInfo V2 is a unified, declarative testing infrastructure that automatically generates comprehensive test coverage for PyTorch operators. Instead of manually writing hundreds of test methods, you define operator metadata once and let OpInfo V2 generate tests automatically.

## Quick Start

### 1. Define Your Operator

Create an `OpInfoV2` definition for your operator:

```python
from torch.testing._internal.opinfo_v2 import (
    OpInfoV2,
    SampleInput,
    TestCategory,
    BASIC_TEST_CATEGORIES,
)

def sample_inputs_my_op(opinfo, device, dtype):
    """Generate sample inputs for testing."""
    yield SampleInput(
        torch.randn(10, device=device, dtype=dtype),
        args=(torch.randn(10, device=device, dtype=dtype),),
        name="basic_case"
    )

my_op = OpInfoV2(
    name='my_op',
    test_categories=BASIC_TEST_CATEGORIES,
    sample_inputs_fn=sample_inputs_my_op,
    supports_autograd=True,
)
```

### 2. Generate Tests

In your test file, generate tests automatically:

```python
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.test_generator import generate_all_tests

# Define your operators
op_db_v2 = [my_op]

# Generate all tests (one line!)
generate_all_tests(globals(), op_db_v2, base_class=TestCase)

if __name__ == '__main__':
    run_tests()
```

That's it! This generates forward, backward, and JIT tests for all supported devices and dtypes.

## OpInfoV2 Schema Reference

### Required Fields

- **`name`** (str): Operator name (e.g., `"add"`, `"relu"`)
- **`sample_inputs_fn`** (Callable): Function that generates test inputs

### Test Categories

Specify which test categories to generate:

```python
from torch.testing._internal.opinfo_v2 import TestCategory

test_categories=[
    TestCategory.FORWARD,      # Basic forward execution
    TestCategory.BACKWARD,     # First-order gradients
    TestCategory.GRADGRAD,     # Second-order gradients
    TestCategory.JIT,          # TorchScript compilation
    TestCategory.VMAP,         # functorch vmap
    # ... and more
]
```

Or use predefined groups:

```python
from torch.testing._internal.opinfo_v2 import (
    BASIC_TEST_CATEGORIES,      # Forward, Backward, JIT
    GRADIENT_TEST_CATEGORIES,   # Forward, Backward, Gradgrad, Forward AD
    ALL_TEST_CATEGORIES,        # All available categories
)
```

### Type Support

Control which dtypes are tested:

```python
from torch.testing._internal.opinfo_v2 import all_types_and_complex_and

OpInfoV2(
    name='my_op',
    # Option 1: Specify exact dtypes
    dtypes=all_types_and_complex_and(torch.bool, torch.half),

    # Option 2: Use boolean flags (if dtypes=None)
    floating_dtypes=True,   # Test float32, float64, float16, bfloat16
    integral_dtypes=True,   # Test int8, int16, int32, int64, uint8
    complex_dtypes=True,    # Test complex64, complex128
)
```

### Device Support

Control which devices are tested:

```python
OpInfoV2(
    name='my_op',
    supports_cpu=True,
    supports_cuda=True,
    supports_mps=True,   # Apple Silicon
    supports_xpu=True,   # Intel XPU
)
```

### Gradient Testing

Configure gradient testing:

```python
OpInfoV2(
    name='my_op',
    supports_autograd=True,      # Enable backward tests
    supports_gradgrad=True,      # Enable second-order gradient tests
    supports_forward_ad=True,    # Enable forward-mode AD tests

    # Custom gradcheck parameters
    gradcheck_kwargs={'atol': 1e-4, 'rtol': 1e-3},
)
```

### Shape Constraints

Specify shape requirements:

```python
OpInfoV2(
    name='conv2d',
    min_ndim=4,              # Requires at least 4D input
    max_ndim=4,              # Requires at most 4D input
)
```

### Memory Format Support

For operators supporting channels-last:

```python
OpInfoV2(
    name='my_op',
    supports_channels_last=True,
)
```

### Reference Implementation

Provide a reference (e.g., NumPy) for validation:

```python
import numpy as np

OpInfoV2(
    name='add',
    reference_fn=lambda x, y: np.add(x, y),
)
```

### Skips and Expected Failures

Skip tests for known issues:

```python
from torch.testing._internal.opinfo_v2 import SkipInfo, FailureInfo

OpInfoV2(
    name='my_op',
    skips=[
        SkipInfo(
            'TestGradients',
            'test_backward',
            dtypes=[torch.int64],
            reason='Gradients not defined for integers',
        ),
    ],
    expected_failures=[
        FailureInfo(
            'TestJIT',
            'test_jit',
            devices=['mps'],
            reason='Known MPS JIT issue',
        ),
    ],
)
```

## Sample Input Functions

Sample input functions generate test cases:

```python
def sample_inputs_my_op(opinfo, device, dtype):
    """Generate sample inputs.

    Args:
        opinfo: The OpInfoV2 instance
        device: Device to create tensors on ('cpu', 'cuda', etc.)
        dtype: Data type for tensors

    Yields:
        SampleInput: Test input cases
    """
    # Basic case
    yield SampleInput(
        torch.randn(10, device=device, dtype=dtype),
        name="basic"
    )

    # With positional arguments
    yield SampleInput(
        torch.randn(10, device=device, dtype=dtype),
        args=(torch.randn(10, device=device, dtype=dtype),),
        name="with_args"
    )

    # With keyword arguments
    yield SampleInput(
        torch.randn(10, device=device, dtype=dtype),
        kwargs={'alpha': 2.0},
        name="with_kwargs"
    )

    # Broadcasting
    yield SampleInput(
        torch.randn(3, 4, device=device, dtype=dtype),
        args=(torch.randn(4, device=device, dtype=dtype),),
        name="broadcast"
    )
```

## Advanced Usage

### Test Optimization

Reduce redundant tests:

```python
from torch.testing._internal.test_optimizer import TestOptimizer

generator = TestGenerator()
optimizer = TestOptimizer(optimization_level='medium')

for opinfo in op_db_v2:
    tests = generator.generate_tests(opinfo)
    optimized = optimizer.filter_redundant_tests(opinfo, tests)
    # Use optimized tests (30-40% fewer)
```

### Coverage Analysis

Analyze test coverage:

```python
from tools.test_dashboard.coverage_analyzer import CoverageAnalyzer

analyzer = CoverageAnalyzer()
report = analyzer.generate_report(op_db_v2)
analyzer.print_report(report)

# Save reports
analyzer.save_report(report, 'coverage.json')
analyzer.generate_html_report(report, 'coverage.html')
```

### Test Sharding

Shard tests for parallel execution:

```python
from torch.testing._internal.test_optimizer import TestSharding

sharding = TestSharding(num_shards=4)
my_shard = sharding.get_shard(all_tests, shard_id=0)
```

### Custom Test Classes

Mix auto-generated and custom tests:

```python
class TestMyOp(TestCase):
    """Custom test class."""

    def test_custom_edge_case(self):
        """Hand-written test for specific edge case."""
        # Your custom test here
        pass

# Also generate standard tests for this class
generate_all_tests(
    {'TestMyOp': TestMyOp},
    op_db_v2,
    base_class=TestCase
)
```

## Best Practices

### 1. Comprehensive Sample Inputs

Provide diverse test cases:

```python
def sample_inputs_good(opinfo, device, dtype):
    # ✓ Different shapes
    yield SampleInput(torch.randn(5, device=device, dtype=dtype))
    yield SampleInput(torch.randn(3, 4, device=device, dtype=dtype))

    # ✓ Broadcasting cases
    yield SampleInput(
        torch.randn(3, 4, device=device, dtype=dtype),
        args=(torch.randn(4, device=device, dtype=dtype),),
    )

    # ✓ Edge cases
    yield SampleInput(torch.tensor([], device=device, dtype=dtype))  # Empty
    yield SampleInput(torch.tensor(3.14, device=device, dtype=dtype))  # 0D

    # ✓ Named cases for debugging
    yield SampleInput(
        torch.randn(10, device=device, dtype=dtype),
        name="regression_issue_12345"
    )
```

### 2. Correct Gradient Flags

```python
# ✓ ReLU has first derivative but discontinuous second derivative
OpInfoV2(
    name='relu',
    supports_autograd=True,
    supports_gradgrad=False,
)

# ✓ Smooth functions support all gradients
OpInfoV2(
    name='sigmoid',
    supports_autograd=True,
    supports_gradgrad=True,
    supports_forward_ad=True,
)
```

### 3. Document Known Issues

```python
OpInfoV2(
    name='my_op',
    skips=[
        SkipInfo(
            'TestJIT',
            'test_jit',
            devices=['xpu'],
            reason='TODO: XPU JIT support not implemented (issue #12345)',
        ),
    ],
)
```

### 4. Use Reference Implementations

```python
# ✓ Validate against NumPy when possible
OpInfoV2(
    name='add',
    reference_fn=lambda x, y: np.add(x, y),
)

# For complex functions, you can skip reference
OpInfoV2(
    name='fft',
    reference_fn=None,  # Complex FFT implementation
)
```

## Migration from OpInfo V1

### Before (OpInfo V1)

```python
# test_ops.py - manually written tests
class TestBinaryOps(TestCase):
    @dtypes(*all_types())
    def test_add_cpu(self, device, dtype):
        # Manual implementation
        pass

    @dtypes(*floating_types())
    def test_add_grad(self, device, dtype):
        # Manual implementation
        pass

    # ... 50+ more test methods
```

### After (OpInfo V2)

```python
# opinfo_v2.py - declarative
op_db_v2 = [
    OpInfoV2(
        name='add',
        sample_inputs_fn=sample_inputs_add,
        test_categories=BASIC_TEST_CATEGORIES,
    ),
]

# test_ops_v2.py - auto-generated
generate_all_tests(globals(), op_db_v2)
```

## Troubleshooting

### "OpInfoV2 must provide sample_inputs_fn"

You must provide a sample input generator:

```python
def sample_inputs_my_op(opinfo, device, dtype):
    yield SampleInput(torch.randn(10, device=device, dtype=dtype))

OpInfoV2(
    name='my_op',
    sample_inputs_fn=sample_inputs_my_op,  # ✓ Required
)
```

### "includes BACKWARD tests but supports_autograd=False"

If you enable gradient tests, set `supports_autograd=True`:

```python
OpInfoV2(
    name='my_op',
    test_categories=[TestCategory.BACKWARD],
    supports_autograd=True,  # ✓ Required for gradient tests
)
```

### Tests are too slow

Use test optimization:

```python
optimizer = TestOptimizer(optimization_level='high')
optimized_tests = optimizer.filter_redundant_tests(opinfo, tests)
```

### Need dtype-specific behavior

Use conditional logic in sample inputs:

```python
def sample_inputs_my_op(opinfo, device, dtype):
    if dtype.is_floating_point:
        yield SampleInput(torch.randn(10, device=device, dtype=dtype))
    else:
        yield SampleInput(torch.randint(0, 10, (10,), device=device, dtype=dtype))
```

## Command-Line Tools

### Coverage Analysis

```bash
# Generate coverage report
python tools/test_dashboard/coverage_analyzer.py \
    --op-db torch/testing/_internal/opinfo_v2_examples.py \
    --output coverage.json \
    --html coverage.html \
    --verbose
```

### Running Tests

```bash
# Run all generated tests
python test/test_ops_v2_example.py

# Run specific operator tests
python test/test_ops_v2_example.py TestAddOpsV2

# Run specific test method
python test/test_ops_v2_example.py TestAddOpsV2.test_forward_add_cpu_float32
```

## Examples

See the following files for complete examples:

- `torch/testing/_internal/opinfo_v2_examples.py` - Example operator definitions
- `test/test_ops_v2_example.py` - Example test file with 10+ usage patterns
- `tools/test_dashboard/coverage_analyzer.py` - Coverage analysis tool

## Further Reading

- RFC-0002: Unified Testing Infrastructure (design document)
- PyTorch Testing Guide: https://pytorch.org/docs/stable/community/contribution_guide.html#writing-tests
- Device-Type Testing: `torch/testing/_internal/common_device_type.py`

## Contributing

When adding a new operator to PyTorch:

1. Create an `OpInfoV2` definition in `torch/testing/_internal/common_methods_invocations.py`
2. Implement `sample_inputs_fn` with comprehensive test cases
3. Set appropriate test categories and flags
4. Run coverage analysis to ensure adequate coverage
5. Add any necessary skips for known issues

For questions or issues, please file a GitHub issue or ask on the PyTorch forums.
